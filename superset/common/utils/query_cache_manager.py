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
import time
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
#
# Timeout and "takeover" – the biggest production risk:
# ---------------------------------------------------------------------------
#   The frontend aborts HTTP requests instantly (via AbortController) when a
#   user changes a dashboard filter.  The corresponding backend threads,
#   however, continue waiting inside inflight_guard until the guard timeout
#   expires.  If that timeout is much larger than the actual query runtime,
#   threads pile up holding DB connections and thread-pool slots for requests
#   that nobody will ever read.
#
#   When the timeout DOES fire, the original code woke ALL N−1 waiting threads
#   simultaneously, causing each of them to retry the query — recreating the
#   thundering herd.
#
#   The "takeover" mechanism (see inflight_guard below) fixes this: the FIRST
#   timeout-waiter to wake up atomically replaces the in-flight event and
#   becomes the new sole executor.  All other timeout-waiters see the new event
#   and wait for the takeover thread to finish instead of executing themselves.
#   This keeps the maximum concurrent executions at 2 (original + takeover)
#   regardless of how many threads are waiting.
#
#   The timeout value defaults to SUPERSET_WEBSERVER_TIMEOUT at runtime so
#   that backend threads do not outlive the HTTP connections they serve.
#   Operators can override it via the QUERY_INFLIGHT_TIMEOUT_S config key.
# ---------------------------------------------------------------------------
# All reads and writes to _inflight_events MUST be done while holding
# _inflight_lock to avoid TOCTOU races.  The lock is intentionally
# short-held: we only hold it for the dictionary operation, then release it
# before any blocking (event.wait) or yielding to caller code.
_inflight_events: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()

#: Fallback timeout (seconds) used when no Flask application context is
#: available (e.g. unit tests that call inflight_guard directly).  In
#: production the runtime value is read from QUERY_INFLIGHT_TIMEOUT_S or
#: SUPERSET_WEBSERVER_TIMEOUT via _get_inflight_timeout().
_INFLIGHT_WAIT_TIMEOUT_S = 60

#: TTL (seconds) for the short-lived error sentinels written to the shared
#: cache backend when the "winner" thread of an inflight_guard fails.
#: Waiting threads read this sentinel and return the cached failure
#: immediately instead of all re-executing the same failing query.  The TTL
#: is kept short so that transient failures (e.g. brief DB unavailability)
#: do not suppress later successful queries for longer than necessary.
_ERROR_SENTINEL_TTL_S = 30


def _emit_stat(stat_key: str, value: float | None = None) -> None:
    """Emit a counter or timing stat, silently ignoring missing app context.

    Parameters
    ----------
    stat_key:
        The metric name to increment or record.
    value:
        When ``None`` (default) a counter (``incr``) is emitted.  When a
        float is supplied a timing value in milliseconds (``timing``) is
        emitted instead.

    This helper is used by :func:`inflight_guard` to expose observability
    metrics that let operators distinguish between two very different
    outcomes that can both produce a "high dedupe hit rate":

    * **Good outcome** – waiters coalesce on a single fast query; wait_ms
      is small and overall dashboard latency drops proportionally.
    * **Bad outcome** – waiters coalesce on a slow query (or one that times
      out and triggers a takeover); wait_ms is large and end-to-end latency
      is dominated by guard wait rather than DB query time.

    Metric reference
    ----------------
    ``inflight_guard.deduped``
        Counter incremented each time a thread coalesces instead of
        executing its own query.  High values = high dedupe hit rate.
    ``inflight_guard.wait_ms``
        Timing (ms) of how long a coalesced thread actually blocked.
        If this is comparable to the DB query duration, the guard is
        adding wait time rather than saving work — investigate why the
        winner is slow (check ``inflight_guard.timeout``).
    ``inflight_guard.timeout``
        Counter incremented when a waiter's ``event.wait()`` call times
        out before the winner finishes.  Non-zero values mean the original
        query is exceeding ``QUERY_INFLIGHT_TIMEOUT_S`` / the webserver
        timeout.  High values combined with high latency → the guard is
        adding wait time; reduce the timeout or optimise the underlying
        query.
    ``inflight_guard.takeover``
        Counter incremented when a timeout-waiter successfully takes over
        execution.  Each takeover means one extra query execution.  In
        steady state this should be near zero; spikes indicate queries that
        routinely exceed the guard timeout.
    """
    try:
        stats_logger = current_app.config["STATS_LOGGER"]
        if value is None:
            stats_logger.incr(stat_key)
        else:
            stats_logger.timing(stat_key, value)
    except RuntimeError:
        # No active Flask application context (e.g. unit tests that call
        # inflight_guard directly without a Flask app).  Safe to ignore.
        pass


def _get_inflight_timeout() -> int:
    """Return the configured inflight-guard wait timeout in seconds.

    Reads ``QUERY_INFLIGHT_TIMEOUT_S`` from the Flask application config when
    a request context is active.  Falls back to ``SUPERSET_WEBSERVER_TIMEOUT``
    when ``QUERY_INFLIGHT_TIMEOUT_S`` is ``None`` (the default), and falls
    back to the module-level ``_INFLIGHT_WAIT_TIMEOUT_S`` constant when there
    is no Flask application context (e.g. unit tests).
    """
    try:
        cfg = current_app.config
        explicit = cfg.get("QUERY_INFLIGHT_TIMEOUT_S")
        if explicit is not None:
            return int(explicit)
        return int(cfg.get("SUPERSET_WEBSERVER_TIMEOUT", _INFLIGHT_WAIT_TIMEOUT_S))
    except RuntimeError:
        # No active Flask application context (unit tests).
        return _INFLIGHT_WAIT_TIMEOUT_S


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

    Timeout and takeover
    --------------------
    The wait timeout is read from the ``QUERY_INFLIGHT_TIMEOUT_S`` Flask
    config key, falling back to ``SUPERSET_WEBSERVER_TIMEOUT``.  This keeps
    the backend wait aligned with the HTTP deadline so threads do not outlive
    the connections they serve.

    When a waiter's timeout fires:

    * The **first** waiter to notice atomically replaces the original event in
      ``_inflight_events`` with a new one and becomes the "takeover" executor,
      yielding ``True`` so the caller runs the query.
    * All **other** waiters see the replacement event and wait for the takeover
      executor to complete, then yield ``False`` so the caller re-reads cache.

    This keeps concurrent executions at most 2 (original + one takeover)
    regardless of the number of waiting threads, preventing the thundering herd
    that would otherwise occur when all N−1 waiters time out simultaneously.

    The original executor (Thread-1) only removes its **own** event from
    ``_inflight_events`` on cleanup, so it cannot accidentally discard the
    takeover thread's replacement event.

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

    timeout = _get_inflight_timeout()

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
        # --- Observability: dedupe hit ----------------------------------------
        # Count every thread that coalesces instead of executing its own query.
        # Pair with wait_ms below to distinguish "guard saved work" (low wait_ms)
        # from "guard just added latency" (wait_ms ≈ full query duration).
        _emit_stat("inflight_guard.deduped")
        _wait_start = time.monotonic()
        was_set = event.wait(timeout=timeout)
        # Emit how long this thread actually blocked regardless of outcome.
        _emit_stat("inflight_guard.wait_ms", (time.monotonic() - _wait_start) * 1000)

        if not was_set:
            # The wait timed out: the original executor is still running.
            # Race all timeout-waiters: the FIRST to win the lock atomically
            # replaces the original event and becomes the new executor.
            # All losers discover the replacement and wait for the winner.
            # --- Observability: timeout ----------------------------------------
            # A non-zero timeout counter means the original query is exceeding
            # QUERY_INFLIGHT_TIMEOUT_S.  High timeouts + high latency → the
            # guard is adding wait time; reduce the timeout or fix the query.
            _emit_stat("inflight_guard.timeout")
            logger.warning(
                "Single-flight: guard timed out after %ss for cache key '%s'. "
                "Original query is still running. "
                "One waiter will take over; others will wait for the takeover.",
                timeout,
                cache_key,
            )
            with _inflight_lock:
                if _inflight_events.get(cache_key) is event:
                    # We are the takeover winner: install a fresh event.
                    takeover_event = threading.Event()
                    _inflight_events[cache_key] = takeover_event
                    is_takeover = True
                else:
                    # Another thread already took over; wait for it.
                    is_takeover = False
                    current_event = _inflight_events.get(cache_key)

            if is_takeover:
                # --- Observability: takeover -----------------------------------
                # Each takeover = one extra query execution beyond the original.
                # In steady state this should be near zero; spikes indicate
                # queries routinely exceeding the guard timeout.
                _emit_stat("inflight_guard.takeover")
                logger.warning(
                    "Single-flight: this thread is taking over execution "
                    "for cache key '%s'.",
                    cache_key,
                )
                try:
                    yield True
                finally:
                    with _inflight_lock:
                        if _inflight_events.get(cache_key) is takeover_event:
                            _inflight_events.pop(cache_key, None)
                    takeover_event.set()
                return

            # Not the takeover winner: wait for the takeover thread.
            if current_event is not None:
                current_event.wait(timeout=timeout)

        yield False
        return

    # We are the first thread for this cache key.
    try:
        yield True
    finally:
        # Only remove OUR event – a concurrent takeover may have already
        # replaced it with a new one that other threads are waiting on.
        with _inflight_lock:
            if _inflight_events.get(cache_key) is event:
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
            # Detect an error sentinel written by a failed winner thread.
            # Setting is_loaded=True signals callers to skip re-execution;
            # status=FAILED and error_message tell them to surface the error.
            if "__error__" in cache_value:
                query_cache.error_message = cache_value["__error__"]
                query_cache.status = QueryStatus.FAILED
                query_cache.is_loaded = True
                query_cache.is_cached = True
                logger.debug(
                    "Inflight guard: error sentinel hit for key %s – "
                    "returning cached failure without re-executing",
                    key,
                )
                return query_cache

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

    @classmethod
    def set_error_sentinel(
        cls,
        key: str,
        error_message: str,
        region: CacheRegion = CacheRegion.DATA,
    ) -> None:
        """Write a short-lived error marker to the shared cache backend.

        Called by the winner thread of ``inflight_guard`` when a query fails,
        so that threads blocked on the same cache key can read the failure
        result immediately instead of all re-executing the same failing query.

        Sentinels are stored as ``{"__error__": <message>}`` and expire after
        ``_ERROR_SENTINEL_TTL_S`` seconds.  The short TTL ensures that
        transient failures (e.g. brief DB unavailability) do not suppress
        later successful queries for longer than necessary.

        Recognised by ``QueryCacheManager.get()`` via the ``"__error__"`` key.
        """
        if key and _cache.get(region):
            cls.set(
                key=key,
                value={"__error__": error_message},
                timeout=_ERROR_SENTINEL_TTL_S,
                region=region,
            )

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
