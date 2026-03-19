# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from flask import current_app
from flask_caching import Cache
from pandas import DataFrame

from superset.common.db_query_status import QueryStatus
from superset.constants import CacheRegion
from superset.exceptions import CacheLoadError
from superset.extensions import cache_manager
from superset.models.helpers import QueryResult
from superset.stats_logger import BaseStatsLogger
from superset.superset_typing import Column
from superset.utils.cache import set_and_log_cache
from superset.utils.core import error_msg_from_exception, get_stacktrace

logger = logging.getLogger(__name__)

_cache: dict[CacheRegion, Cache] = {
    CacheRegion.DEFAULT: cache_manager.cache,
    CacheRegion.DATA: cache_manager.data_cache,
}

# ---------------------------------------------------------------------------
# Single-flight / in-flight deduplication
# ---------------------------------------------------------------------------
# These module-level structures provide *within-process* single-flight
# protection for query execution.
#
# Problem this solves (thundering-herd / dogpile):
#   When a dashboard loads with N charts that share the same datasource and
#   filters, each chart independently POSTs to /api/v1/chart/data.  All N
#   requests reach get_df_payload() at roughly the same time and see a cache
#   miss (the first request hasn't finished yet).  Without a guard they all
#   execute the identical SQL query against the database.
#
# How it works:
# ---------------------------------------------------------------------------
#   The first thread to see a cache miss for a given cache_key acquires
#   "ownership" by inserting a threading.Event into _inflight_events and
#   yields is_first=True.  Subsequent threads that want the same key find the
#   Event already there, yield is_first=False, and block on event.wait().
#   Once the first thread completes (success *or* failure) it fires the Event
#   so that all waiters are unblocked and can re-read from cache.
#
# Security model – how per-user isolation is maintained:
# ---------------------------------------------------------------------------
#   The guard key is the *security-scoped* cache key produced by
#   QueryContextProcessor.query_cache_key(), which already encodes:
#     • Row-Level Security predicates for the current user (via
#       security_manager.get_rls_cache_key()), including guest-token RLS
#     • Jinja template user-context (current_username(), current_user_id(),
#       url_param(), etc.) resolved at request time and appended via
#       datasource.get_extra_cache_keys()
#     • Database-level impersonation identity when CACHE_IMPERSONATION,
#       CACHE_QUERY_BY_USER, or per_user_caching options are active
#
#   Consequence: two users with *different* effective permissions produce
#   different cache keys → different guard keys → they never block each other
#   and never share cached results.  The guard coalesces requests only when
#   the callers would see the exact same data, making deduplication safe.
#
# Scope and limitations:
#   * Only protects threads within the same worker process (e.g. Gunicorn
#     threaded or gevent worker).  Across separate processes the shared cache
#     (Redis / Memcached) already provides eventual consistency – the second
#     process to finish a query will simply overwrite the cached value with an
#     identical result, which is harmless.
#   * force_query=True bypasses the guard intentionally: the user has
#     explicitly requested fresh data, so we should not coalesce that request.
# ---------------------------------------------------------------------------
# All reads and writes to _inflight_events MUST be done while holding
# _inflight_lock to avoid TOCTOU races.  The lock is intentionally
# short-held: we only hold it for the dictionary operation, then release it
# before any blocking (event.wait) or yielding to caller code.
_inflight_events: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()

#: How long (seconds) a waiter will block before giving up and executing the
#: query itself.  Acts as a safety valve so a slow/hung query never locks
#: other threads indefinitely.  Should be set conservatively higher than the
#: expected worst-case query duration.  Not currently user-configurable, but
#: can be overridden in tests by monkey-patching this module attribute.
_INFLIGHT_WAIT_TIMEOUT_S = 60


@contextmanager
def inflight_guard(
    cache_key: str | None,
) -> Generator[bool, None, None]:
    """Single-flight context manager for query execution.

    Yields ``True`` (is_first) when the caller should execute the query.
    Yields ``False`` when another thread is already executing the same query;
    in this case the caller should wait (the wait happens inside the manager)
    and then re-check the cache rather than executing a redundant query.

    When *cache_key* is ``None`` (e.g. force_query mode or no key available)
    the guard is a no-op and always yields ``True`` so the caller executes
    unconditionally.

    Security contract
    -----------------
    The guard coalesces concurrent threads that share the **exact same**
    *cache_key*.  It is the **caller's responsibility** to pass a key that
    already encodes all user-specific security context so that two threads
    representing users with different data-access rights never share a key.

    In practice this means callers must use the key produced by
    ``QueryContextProcessor.query_cache_key()``, which folds in:

    * **RLS predicates** – via ``security_manager.get_rls_cache_key()``,
      resolved for the *current* Flask-request user.  A user with no RLS rules
      gets an empty list; a user with RLS rules gets the list of applicable
      filter clauses.  Different lists → different keys → different guards.
    * **Jinja user-context** – via ``datasource.get_extra_cache_keys()``.
      Template functions such as ``current_username()``, ``current_user_id()``,
      and ``url_param()`` are evaluated at request time and appended to the
      key, differentiating per-user filtered virtual datasets.
    * **Impersonation identity** – when ``CACHE_IMPERSONATION``,
      ``CACHE_QUERY_BY_USER``, or ``per_user_caching`` flags are active, the
      database-level username is included so that impersonated sessions produce
      distinct cache buckets.

    Because of these inclusions, two requests for the same raw payload but
    with different effective permissions will produce different cache keys and
    therefore use independent guards.  The guard will **never** cause a
    restricted user to receive data that was fetched in the context of a less-
    restricted user.

    Usage::

        with inflight_guard(cache_key) as is_first:
            if not is_first:
                cache = QueryCacheManager.get(key=cache_key, ...)
            if is_first or not cache.is_loaded:
                query_result = run_expensive_query()
                cache.set_query_result(key=cache_key, ...)
    """
    if not cache_key:
        yield True
        return

    with _inflight_lock:
        if cache_key in _inflight_events:
            event = _inflight_events[cache_key]
            is_first = False
        else:
            event = threading.Event()
            _inflight_events[cache_key] = event
            is_first = True

    if not is_first:
        logger.debug(
            "Single-flight: waiting for in-flight query with cache key: %s", cache_key
        )
        event.wait(timeout=_INFLIGHT_WAIT_TIMEOUT_S)
        yield False
        return

    # We are the first thread for this cache key.
    try:
        yield True
    finally:
        # Unblock all waiters and remove our entry regardless of outcome.
        with _inflight_lock:
            _inflight_events.pop(cache_key, None)
        event.set()


class QueryCacheManager:
    """
    Class for manage query-cache getting and setting
    """

    @property
    def stats_logger(self) -> BaseStatsLogger:
        return current_app.config["STATS_LOGGER"]

    # pylint: disable=too-many-instance-attributes,too-many-arguments
    def __init__(
        self,
        df: DataFrame = DataFrame(),  # noqa: B008
        query: str = "",
        annotation_data: dict[str, Any] | None = None,
        applied_template_filters: list[str] | None = None,
        applied_filter_columns: list[Column] | None = None,
        rejected_filter_columns: list[Column] | None = None,
        status: str | None = None,
        error_message: str | None = None,
        is_loaded: bool = False,
        stacktrace: str | None = None,
        is_cached: bool | None = None,
        cache_dttm: str | None = None,
        cache_value: dict[str, Any] | None = None,
        sql_rowcount: int | None = None,
        queried_dttm: str | None = None,
    ) -> None:
        self.df = df
        self.query = query
        self.annotation_data = {} if annotation_data is None else annotation_data
        self.applied_template_filters = applied_template_filters or []
        self.applied_filter_columns = applied_filter_columns or []
        self.rejected_filter_columns = rejected_filter_columns or []
        self.status = status
        self.error_message = error_message

        self.is_loaded = is_loaded
        self.stacktrace = stacktrace
        self.is_cached = is_cached
        self.cache_dttm = cache_dttm
        self.cache_value = cache_value
        self.sql_rowcount = sql_rowcount
        self.queried_dttm = queried_dttm

    # pylint: disable=too-many-arguments
    def set_query_result(
        self,
        key: str,
        query_result: QueryResult,
        annotation_data: dict[str, Any] | None = None,
        force_query: bool | None = False,
        timeout: int | None = None,
        datasource_uid: str | None = None,
        region: CacheRegion = CacheRegion.DEFAULT,
    ) -> None:
        """
        Set dataframe of query-result to specific cache region
        """
        try:
            self.status = query_result.status
            self.query = query_result.query
            self.applied_template_filters = query_result.applied_template_filters
            self.applied_filter_columns = query_result.applied_filter_columns
            self.rejected_filter_columns = query_result.rejected_filter_columns
            self.error_message = query_result.error_message
            self.df = query_result.df
            self.sql_rowcount = query_result.sql_rowcount
            self.annotation_data = {} if annotation_data is None else annotation_data
            self.queried_dttm = (
                datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
            )

            if self.status != QueryStatus.FAILED:
                current_app.config["STATS_LOGGER"].incr("loaded_from_source")
                if not force_query:
                    current_app.config["STATS_LOGGER"].incr(
                        "loaded_from_source_without_force"
                    )
                self.is_loaded = True

            value = {
                "df": self.df,
                "query": self.query,
                "applied_template_filters": self.applied_template_filters,
                "applied_filter_columns": self.applied_filter_columns,
                "rejected_filter_columns": self.rejected_filter_columns,
                "annotation_data": self.annotation_data,
                "sql_rowcount": self.sql_rowcount,
                "queried_dttm": self.queried_dttm,
                "dttm": self.queried_dttm,  # Backwards compatibility
            }
            if self.is_loaded and key and self.status != QueryStatus.FAILED:
                self.set(
                    key=key,
                    value=value,
                    timeout=timeout,
                    datasource_uid=datasource_uid,
                    region=region,
                )
        except Exception as ex:  # pylint: disable=broad-except
            logger.exception(ex)
            if not self.error_message:
                self.error_message = str(ex)
            self.status = QueryStatus.FAILED
            self.stacktrace = get_stacktrace()

    @classmethod
    def get(
        cls,
        key: str | None,
        region: CacheRegion = CacheRegion.DEFAULT,
        force_query: bool | None = False,
        force_cached: bool | None = False,
    ) -> QueryCacheManager:
        """
        Initialize QueryCacheManager by query-cache key
        """
        query_cache = cls()
        if not key or not _cache[region] or force_query:
            return query_cache

        if cache_value := _cache[region].get(key):
            logger.debug("Cache key: %s", key)
            # Log cache hit for debugging
            logger.debug("CACHE GET - Key: %s, Region: %s", key, region)
            current_app.config["STATS_LOGGER"].incr("loading_from_cache")
            try:
                query_cache.df = cache_value["df"]
                query_cache.query = cache_value["query"]
                query_cache.annotation_data = cache_value.get("annotation_data", {})
                query_cache.applied_template_filters = cache_value.get(
                    "applied_template_filters", []
                )
                query_cache.applied_filter_columns = cache_value.get(
                    "applied_filter_columns", []
                )
                query_cache.rejected_filter_columns = cache_value.get(
                    "rejected_filter_columns", []
                )
                query_cache.status = QueryStatus.SUCCESS
                query_cache.is_loaded = True
                query_cache.is_cached = cache_value is not None
                query_cache.sql_rowcount = cache_value.get("sql_rowcount", None)
                query_cache.cache_dttm = (
                    cache_value["dttm"] if cache_value is not None else None
                )
                query_cache.queried_dttm = cache_value.get(
                    "queried_dttm", cache_value.get("dttm")
                )
                query_cache.cache_value = cache_value
                current_app.config["STATS_LOGGER"].incr("loaded_from_cache")
            except KeyError as ex:
                logger.exception(ex)
                logger.error(
                    "Error reading cache: %s",
                    error_msg_from_exception(ex),
                    exc_info=True,
                )
            logger.debug("Serving from cache")

        if force_cached and not query_cache.is_loaded:
            logger.warning(
                "force_cached (QueryContext): value not found for key %s", key
            )
            raise CacheLoadError("Error loading data from cache")
        return query_cache

    @staticmethod
    def set(
        key: str | None,
        value: dict[str, Any],
        timeout: int | None = None,
        datasource_uid: str | None = None,
        region: CacheRegion = CacheRegion.DEFAULT,
    ) -> None:
        """
        set value to specify cache region, proxy for `set_and_log_cache`
        """
        if key:
            set_and_log_cache(_cache[region], key, value, timeout, datasource_uid)

    @staticmethod
    def delete(
        key: str | None,
        region: CacheRegion = CacheRegion.DEFAULT,
    ) -> None:
        if key:
            _cache[region].delete(key)

    @staticmethod
    def has(
        key: str | None,
        region: CacheRegion = CacheRegion.DEFAULT,
    ) -> bool:
        return bool(_cache[region].get(key)) if key else False
