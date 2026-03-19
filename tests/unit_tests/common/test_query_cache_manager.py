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
"""
Tests for QueryCacheManager and the inflight_guard single-flight mechanism.

Background – where duplicate queries were occurring
----------------------------------------------------
When a dashboard with N charts that share the same datasource/filters loads,
each chart independently POSTs to /api/v1/chart/data.  All N requests reach
``get_df_payload()`` at roughly the same time and observe a cache miss (the
first request has not finished yet and nothing is in cache).  Without any
coordination they ALL execute the identical SQL query against the database,
causing:

  * N×  database load instead of 1×
  * N×  serialisation of identical results
  * A "thundering herd" / dogpile effect on every dashboard page-load

The ``inflight_guard`` context manager in ``query_cache_manager.py`` fixes
this by tracking in-flight queries inside a single worker process.  The first
thread acquires ownership of a cache key and executes the query.  Every
subsequent thread that wants the same key blocks until the first thread
signals completion and then re-reads the result from cache.
"""

import threading

import pytest

from superset.common.utils.query_cache_manager import (
    _inflight_events,
    _inflight_lock,
    inflight_guard,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_inflight() -> None:
    """Clear the module-level in-flight tracking dict between tests."""
    with _inflight_lock:
        _inflight_events.clear()


@pytest.fixture(autouse=True)
def clean_inflight():
    """Ensure no leftover in-flight state from a previous test."""
    _clear_inflight()
    yield
    _clear_inflight()


# ---------------------------------------------------------------------------
# Basic behaviour tests
# ---------------------------------------------------------------------------


def test_inflight_guard_no_key_always_yields_true():
    """When cache_key is None the guard is a no-op and yields is_first=True."""
    with inflight_guard(None) as is_first:
        assert is_first is True


def test_inflight_guard_empty_string_key_always_yields_true():
    """An empty string is falsy and treated like None (no-op)."""
    with inflight_guard("") as is_first:
        assert is_first is True


def test_inflight_guard_first_caller_yields_true():
    """The first caller for a new cache key should receive is_first=True."""
    with inflight_guard("key-A") as is_first:
        assert is_first is True


def test_inflight_guard_cleans_up_after_exit():
    """The in-flight dict should be empty after the context manager exits."""
    with inflight_guard("key-B"):
        pass
    with _inflight_lock:
        assert "key-B" not in _inflight_events


def test_inflight_guard_reusable_after_completion():
    """A key can be used again once the first execution has completed."""
    with inflight_guard("key-C") as is_first:
        assert is_first is True
    # Second call should also see is_first=True since the first is done.
    with inflight_guard("key-C") as is_first:
        assert is_first is True


def test_inflight_guard_cleans_up_on_exception():
    """Even if an exception is raised inside, the guard must clean up."""
    with pytest.raises(RuntimeError):
        with inflight_guard("key-err"):
            raise RuntimeError("boom")
    with _inflight_lock:
        assert "key-err" not in _inflight_events


# ---------------------------------------------------------------------------
# Concurrency / single-flight tests
# ---------------------------------------------------------------------------


def test_inflight_guard_second_thread_waits_and_reuses():
    """
    Core single-flight test.

    Sequence:
      1. Thread-1 enters inflight_guard, sees is_first=True, starts "executing".
      2. Thread-2 enters inflight_guard for the same key before Thread-1 exits.
         It must block (is_first=False) and wait.
      3. Thread-1 finishes and exits the context (fires the Event).
      4. Thread-2 unblocks and receives is_first=False.

    This mirrors what happens when N dashboard charts POST concurrently for
    the same datasource/query.
    """
    key = "concurrent-key"
    results: list[bool] = []
    errors: list[str] = []

    # Barrier so both threads enter inflight_guard before either exits.
    thread1_inside = threading.Event()
    thread2_done = threading.Event()

    def thread1_fn() -> None:
        with inflight_guard(key) as is_first:
            results.append(is_first)
            # Signal that we're inside the guard (holding the in-flight slot).
            thread1_inside.set()
            # Wait until thread-2 has had a chance to observe the in-flight state
            # and block.  Give it up to 5 seconds.
            thread2_done.wait(timeout=5)

    def thread2_fn() -> None:
        # Wait until thread-1 is inside its guard before we start.
        thread1_inside.wait(timeout=5)
        with inflight_guard(key) as is_first:
            results.append(is_first)
        thread2_done.set()

    t1 = threading.Thread(target=thread1_fn)
    t2 = threading.Thread(target=thread2_fn)

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    if errors:
        pytest.fail("\n".join(errors))

    assert len(results) == 2, f"Expected 2 results, got {results}"
    # Thread-1 should be first (True), Thread-2 should be second (False).
    assert results[0] is True, "Thread-1 should receive is_first=True"
    assert results[1] is False, "Thread-2 should receive is_first=False"


def test_only_one_query_executed_for_concurrent_requests():
    """
    Simulates the thundering-herd scenario at the level of inflight_guard.

    This is the core problem that was occurring:
      * Dashboard loads with N charts that share the same datasource+filters.
      * Each chart independently POSTs to /api/v1/chart/data.
      * All N threads reach the cache-miss branch of get_df_payload concurrently.
      * WITHOUT inflight_guard: all N threads execute the SQL query.
      * WITH inflight_guard: only the first thread executes; the rest wait and
        then find the result in the in-memory "cache" populated by thread-1.

    Rather than wiring up the full Flask stack we model the critical section
    directly:  a shared dict acts as the cache, and inflight_guard is what we
    are verifying.
    """
    concurrency = 5
    key = "dashboard-shared-query"
    fake_cache: dict[str, str] = {}
    exec_count = 0
    exec_lock = threading.Lock()
    start_barrier = threading.Barrier(concurrency)
    errors: list[str] = []

    def simulate_query_with_guard() -> None:
        """Mirrors the critical section inside get_df_payload."""
        nonlocal exec_count
        try:
            # Phase 1: read cache (all threads see a miss at start)
            cache_hit = key in fake_cache

            if not cache_hit:
                # Phase 2: single-flight guard
                with inflight_guard(key) as is_first:
                    if not is_first:
                        # Re-read after waiting for the first thread
                        cache_hit = key in fake_cache

                    if is_first or not cache_hit:
                        # Simulate DB query (slow operation)
                        with exec_lock:
                            exec_count += 1
                        # Store in "cache"
                        fake_cache[key] = "result"
        except (RuntimeError, threading.BrokenBarrierError) as exc:
            # Capture any unexpected thread failures so the main thread can
            # report them via pytest.fail() rather than silently swallowing.
            errors.append(f"{threading.current_thread().name}: {exc}")

    threads = [
        threading.Thread(
            target=lambda: (start_barrier.wait(), simulate_query_with_guard())
        )  # noqa: E501
        for _ in range(concurrency)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    if errors:
        pytest.fail(f"Thread errors: {errors}")

    assert exec_count == 1, (
        f"Expected exactly 1 query execution with single-flight guard, "
        f"but {exec_count} were executed (thundering-herd not prevented)"
    )
    assert fake_cache[key] == "result", "Cache should be populated"
