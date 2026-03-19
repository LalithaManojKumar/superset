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

Security model
--------------
The guard coalesces only threads that share the **exact same** cache key.
``QueryContextProcessor.query_cache_key()`` folds user-security context into
the key so that users with different effective permissions always produce
different keys:

* ``rls=security_manager.get_rls_cache_key(datasource)`` – appends the
  Row-Level Security predicates applicable to the *current* user.  A user
  with no RLS gets an empty list; a user with RLS rules gets the filter-clause
  strings.  Different RLS → different key → independent guard entry.
* ``extra_cache_keys=datasource.get_extra_cache_keys(...)`` – resolves Jinja
  template calls such as ``{{ current_username() }}`` in datasource SQL.
* Impersonation key – added when ``CACHE_IMPERSONATION`` /
  ``CACHE_QUERY_BY_USER`` / ``per_user_caching`` flags are active.

Tests in the "Security isolation" section below verify that:
  1. Different cache keys (representing different user permissions) never block
     each other and never share guard state.
  2. Concurrent threads with the same cache key (same effective permissions,
     safe to deduplicate) are correctly serialised by the guard.
"""

import threading
import time

import pytest

import superset.common.utils.query_cache_manager as query_cache_module
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


# ---------------------------------------------------------------------------
# Security isolation tests
#
# These tests verify that the inflight_guard preserves per-user data
# isolation.  The guard is safe because it keys on the *security-scoped*
# cache key produced by QueryContextProcessor.query_cache_key(), which
# encodes Row-Level Security predicates, Jinja user-context, and
# (optionally) database-level impersonation identity.
#
# Key properties verified:
#   1. Different cache keys (representing different user permissions) have
#      completely independent guard entries – they never block each other.
#   2. Users who share a cache key (same effective permissions) are correctly
#      deduplicated by the guard, and the "waiter" correctly re-reads the
#      result rather than executing a second query.
#   3. Cross-key contamination is impossible: a waiter on key A is never
#      unblocked by an event fired for key B.
# ---------------------------------------------------------------------------


def test_different_security_keys_are_fully_independent():
    """
    Users with different effective permissions produce different cache keys.
    Guards for different keys must operate completely independently – a thread
    holding key-A must NOT block or influence a thread using key-B.

    Real-world mapping:
    ─────────────────
    key_admin  →  admin user, no RLS  → rls=[]  → key hash includes []
    key_user   →  restricted user, RLS applied  → rls=["dept='sales'"]  → different hash

    Both users should be able to execute their queries concurrently without
    either blocking the other.
    """
    # Simulate two different users' security-scoped cache keys.
    # In production these would differ because security_manager.get_rls_cache_key()
    # returns different clause lists for different users.
    key_admin = "cache_key:ds=1:rls=[]"  # admin – no RLS predicates
    key_user = "cache_key:ds=1:rls=[dept=sales]"  # restricted – with RLS

    admin_is_first_values: list[bool] = []
    user_is_first_values: list[bool] = []
    both_inside = threading.Barrier(2)  # ensures real concurrency

    def admin_thread() -> None:
        with inflight_guard(key_admin) as is_first:
            admin_is_first_values.append(is_first)
            both_inside.wait(timeout=5)

    def restricted_thread() -> None:
        with inflight_guard(key_user) as is_first:
            user_is_first_values.append(is_first)
            both_inside.wait(timeout=5)

    t_admin = threading.Thread(target=admin_thread)
    t_user = threading.Thread(target=restricted_thread)

    t_admin.start()
    t_user.start()
    t_admin.join(timeout=10)
    t_user.join(timeout=10)

    # Both should see is_first=True because their keys are different.
    # Neither should have been blocked by the other.
    assert admin_is_first_values == [True], (
        "Admin thread should execute independently (is_first=True)"
    )
    assert user_is_first_values == [True], (
        "Restricted-user thread should execute independently (is_first=True)"
    )


def test_same_security_key_deduplicates_safely():
    """
    Users who share a cache key have identical effective permissions and are
    entitled to see the same data.  The guard should deduplicate their queries:
    the first thread executes, the second waits and reuses the result.

    Real-world mapping:
    ─────────────────
    Both requests belong to users whose RLS predicates produce the same clause
    list (e.g. two users in the same restricted role).  query_cache_key() will
    return the same hash for both, so deduplication is both correct and safe.
    """
    shared_key = "cache_key:ds=1:rls=[region=EU]"
    fake_cache: dict[str, str] = {}
    exec_count = 0
    exec_lock = threading.Lock()

    thread1_inside = threading.Event()
    thread2_done = threading.Event()

    def user1_fn() -> None:
        nonlocal exec_count
        cache_hit = shared_key in fake_cache
        if not cache_hit:
            with inflight_guard(shared_key) as is_first:
                if not is_first:
                    # Re-read after waiting – safe because same permissions
                    cache_hit = shared_key in fake_cache
                if is_first or not cache_hit:
                    with exec_lock:
                        exec_count += 1
                    fake_cache[shared_key] = "EU-result"
                thread1_inside.set()
                thread2_done.wait(timeout=5)

    def user2_fn() -> None:
        nonlocal exec_count
        thread1_inside.wait(timeout=5)  # ensure user1 is inside its guard
        cache_hit = shared_key in fake_cache
        if not cache_hit:
            with inflight_guard(shared_key) as is_first:
                if not is_first:
                    cache_hit = shared_key in fake_cache
                if is_first or not cache_hit:
                    with exec_lock:
                        exec_count += 1
                    fake_cache[shared_key] = "EU-result"
        thread2_done.set()

    t1 = threading.Thread(target=user1_fn)
    t2 = threading.Thread(target=user2_fn)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert exec_count == 1, (
        f"Same-permissions users should share a single query execution, "
        f"but {exec_count} executions occurred"
    )
    assert fake_cache[shared_key] == "EU-result"


def test_different_key_threads_never_unblock_each_other():
    """
    An Event fired for key-A must NOT unblock a waiter on key-B.

    This verifies there is no cross-key contamination: the guard stores one
    threading.Event per cache key, so signalling key-A's event leaves key-B's
    event untouched.
    """
    key_a = "key-user-a"
    key_b = "key-user-b"

    # key_b waiter thread state
    key_b_is_first_received: list[bool] = []
    key_b_entered_guard = threading.Event()
    key_b_allowed_to_exit = threading.Event()

    def key_b_first_thread() -> None:
        """Hold the key-B guard slot open."""
        with inflight_guard(key_b) as is_first:
            key_b_is_first_received.append(is_first)
            key_b_entered_guard.set()
            key_b_allowed_to_exit.wait(timeout=10)

    def key_b_waiter_thread() -> None:
        """Wait for key-B to be released."""
        key_b_entered_guard.wait(timeout=5)
        with inflight_guard(key_b) as is_first:
            key_b_is_first_received.append(is_first)

    # key_a fires independently – it must NOT wake up key_b's waiter
    key_a_fired = threading.Event()

    def key_a_thread() -> None:
        with inflight_guard(key_a):
            pass  # complete immediately
        key_a_fired.set()

    t_b_first = threading.Thread(target=key_b_first_thread)
    t_b_waiter = threading.Thread(target=key_b_waiter_thread)
    t_a = threading.Thread(target=key_a_thread)

    t_b_first.start()
    key_b_entered_guard.wait(timeout=5)

    t_b_waiter.start()
    # Give the waiter a moment to enter its guard and block
    time.sleep(0.05)

    # Fire key-A's event – key-B's waiter must remain blocked
    t_a.start()
    key_a_fired.wait(timeout=5)

    # key-B's waiter should still be waiting (not yet unblocked)
    # We verify by checking that the waiter thread has not exited yet
    t_b_waiter.join(timeout=0.1)
    assert t_b_waiter.is_alive(), (
        "key-B waiter was prematurely unblocked by key-A's Event signal "
        "(cross-key contamination detected)"
    )

    # Now release key-B's first thread
    key_b_allowed_to_exit.set()
    t_b_first.join(timeout=5)
    t_b_waiter.join(timeout=5)
    t_a.join(timeout=5)

    # key-B should have had exactly 2 entries: first=True, waiter=False
    assert key_b_is_first_received == [True, False], (
        f"Expected [True, False] for key-B, got {key_b_is_first_received}"
    )


# ---------------------------------------------------------------------------
# Timeout / takeover tests
#
# These tests verify the "takeover" mechanism that prevents the thundering
# herd when the guard wait timeout fires:
#
#   * When the original executor (Thread-1) takes longer than the configured
#     timeout, ALL waiting threads' event.wait() calls return at the same
#     moment.  WITHOUT the takeover mechanism they would ALL retry the query
#     simultaneously, recreating the thundering herd.
#
#   * WITH the takeover mechanism, exactly ONE of the timeout-waiters atomically
#     replaces the in-flight event and becomes the new executor (is_first=True).
#     All other timeout-waiters see the replacement and wait for that single new
#     executor before yielding is_first=False.
#
# The tests monkey-patch the module-level _INFLIGHT_WAIT_TIMEOUT_S so they
# can trigger a short, controllable timeout without actually waiting 60 s.
# ---------------------------------------------------------------------------


def test_timeout_triggers_single_takeover_not_thundering_herd():
    """
    Core takeover test.

    When the guard wait times out, exactly ONE waiter should take over as the
    new executor (is_first=True).  All other waiters must block until the
    takeover finishes, then receive is_first=False.

    Real-world mapping:
    ──────────────────
    Thread-1 is running a slow query (>_INFLIGHT_WAIT_TIMEOUT_S seconds).
    Threads 2-N all enter inflight_guard for the same key and block.
    After the timeout fires, only Thread-2 (the takeover winner) should
    execute.  Threads 3-N must wait for Thread-2, not execute themselves.
    """
    key = "slow-query-key"
    concurrency = 5  # Threads 2-5 are waiters; Thread-1 is the slow original

    # Patch the module timeout to 0.05 s so the test runs fast.
    original_timeout = query_cache_module._INFLIGHT_WAIT_TIMEOUT_S
    query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = 0.05
    try:
        takeover_count = 0  # threads that received is_first=True via takeover
        waiter_count = 0  # threads that received is_first=False after takeover
        counts_lock = threading.Lock()

        # Thread-1 holds the slot for longer than the timeout.
        thread1_inside = threading.Event()
        thread1_allowed_to_exit = threading.Event()

        def slow_first_thread() -> None:
            with inflight_guard(key) as is_first:
                assert is_first is True
                thread1_inside.set()
                # Stay inside past the waiters' timeout
                thread1_allowed_to_exit.wait(timeout=5)

        def waiter_thread() -> None:
            nonlocal takeover_count, waiter_count
            thread1_inside.wait(timeout=5)
            with inflight_guard(key) as is_first:
                with counts_lock:
                    if is_first:
                        takeover_count += 1
                    else:
                        waiter_count += 1

        t1 = threading.Thread(target=slow_first_thread)
        waiters = [
            threading.Thread(target=waiter_thread) for _ in range(concurrency - 1)
        ]

        t1.start()
        thread1_inside.wait(timeout=5)
        for t in waiters:
            t.start()

        # Wait for all waiter threads to finish (the timeout + takeover completes).
        for t in waiters:
            t.join(timeout=10)

        # Now let Thread-1 finish (it should NOT disrupt the waiter results).
        thread1_allowed_to_exit.set()
        t1.join(timeout=5)

        assert takeover_count == 1, (
            f"Exactly ONE timeout-waiter should become the takeover executor, "
            f"but {takeover_count} did (thundering herd on timeout not fixed)"
        )
        assert waiter_count == concurrency - 2, (
            f"All other waiters should receive is_first=False (re-read from cache), "
            f"but {waiter_count} did (expected {concurrency - 2})"
        )
    finally:
        query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = original_timeout


def test_original_executor_cleanup_does_not_discard_takeover_event():
    """
    Thread-1's cleanup must NOT remove the takeover thread's event.

    When Thread-1 finishes *after* a takeover has already replaced its event
    in _inflight_events, Thread-1's finally block should leave the takeover
    event intact so that other threads waiting on it are not orphaned.
    """
    key = "takeover-cleanup-key"

    # Patch timeout to 0.05 s.
    original_timeout = query_cache_module._INFLIGHT_WAIT_TIMEOUT_S
    query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = 0.05
    try:
        thread1_inside = threading.Event()
        thread1_allowed_to_exit = threading.Event()
        takeover_inside = threading.Event()
        takeover_allowed_to_exit = threading.Event()

        def slow_thread1() -> None:
            with inflight_guard(key):
                thread1_inside.set()
                thread1_allowed_to_exit.wait(timeout=10)

        def takeover_thread() -> None:
            thread1_inside.wait(timeout=5)
            with inflight_guard(key) as is_first:
                # We should be the takeover (is_first=True after timeout)
                if is_first:
                    takeover_inside.set()
                    takeover_allowed_to_exit.wait(timeout=10)

        t1 = threading.Thread(target=slow_thread1)
        t_takeover = threading.Thread(target=takeover_thread)

        t1.start()
        thread1_inside.wait(timeout=5)
        t_takeover.start()

        # Wait for timeout to fire and takeover to begin
        takeover_inside.wait(timeout=3)

        # Now let Thread-1 finish while the takeover is still running.
        # Thread-1's cleanup should NOT remove the takeover's event.
        thread1_allowed_to_exit.set()
        t1.join(timeout=5)

        # The takeover event should still be in the dict (Thread-1 must not
        # have removed it).
        with _inflight_lock:
            event_present = key in _inflight_events
        assert event_present, (
            "Thread-1's cleanup removed the takeover thread's event from "
            "_inflight_events, which would orphan any threads waiting on it"
        )

        # Let the takeover finish normally.
        takeover_allowed_to_exit.set()
        t_takeover.join(timeout=5)

        # After takeover completes, the dict entry should be gone.
        with _inflight_lock:
            assert key not in _inflight_events, (
                "Takeover thread should have removed its event on exit"
            )
    finally:
        query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = original_timeout


def test_configurable_timeout_via_module_attribute():
    """
    Verify that _INFLIGHT_WAIT_TIMEOUT_S controls how long waiters block.

    In tests without a Flask app context, _get_inflight_timeout() falls back
    to the module-level _INFLIGHT_WAIT_TIMEOUT_S constant.  Monkey-patching
    that constant must change the actual wait duration seen by waiters.
    """
    key = "timeout-config-key"
    short_timeout = 0.05  # 50 ms

    original_timeout = query_cache_module._INFLIGHT_WAIT_TIMEOUT_S
    query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = short_timeout
    thread1_inside = threading.Event()
    thread1_allowed_to_exit = threading.Event()
    waiter_wait_duration: list[float] = []
    t1 = None
    tw = None
    try:

        def long_running_thread() -> None:
            with inflight_guard(key):
                thread1_inside.set()
                thread1_allowed_to_exit.wait(timeout=10)

        def timing_waiter() -> None:
            thread1_inside.wait(timeout=5)
            start = time.monotonic()
            with inflight_guard(key):
                pass
            waiter_wait_duration.append(time.monotonic() - start)

        t1 = threading.Thread(target=long_running_thread)
        tw = threading.Thread(target=timing_waiter)
        t1.start()
        thread1_inside.wait(timeout=5)
        tw.start()
        tw.join(timeout=5)

        assert waiter_wait_duration, "Waiter thread did not complete"
        # The waiter should have been unblocked close to the short timeout,
        # not after the default 60 s.
        assert waiter_wait_duration[0] < 5, (
            f"Waiter blocked for {waiter_wait_duration[0]:.2f}s, "
            f"expected ~{short_timeout}s — configurable timeout not respected"
        )
    finally:
        # Ensure Thread-1 is always released so it does not block teardown.
        thread1_allowed_to_exit.set()
        if t1 is not None:
            t1.join(timeout=5)
        if tw is not None:
            tw.join(timeout=5)
        query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = original_timeout


# ---------------------------------------------------------------------------
# Observability / metrics tests
#
# These tests verify that inflight_guard emits the correct stats-logger
# counters and timings so operators can debug whether deduplication is
# actually reducing work (low wait_ms) vs just adding latency (high wait_ms).
#
# Metric reference (see _emit_stat docstring for full details):
#   inflight_guard.deduped   – counter; each coalesced waiter thread
#   inflight_guard.wait_ms   – timing; how long each waiter actually blocked
#   inflight_guard.timeout   – counter; guard wait timed out before winner finished
#   inflight_guard.takeover  – counter; a timeout-waiter became the new executor
# ---------------------------------------------------------------------------


def _make_fake_stats_logger():
    """Return a minimal stats-logger stub that records calls."""

    class _FakeStats:
        def __init__(self) -> None:
            self.incr_calls: list[str] = []
            self.timing_calls: list[tuple[str, float]] = []

        def incr(self, key: str) -> None:
            self.incr_calls.append(key)

        def timing(self, key: str, value: float) -> None:
            self.timing_calls.append((key, value))

        def gauge(self, key: str, value: float) -> None:  # noqa: ARG002
            pass

    return _FakeStats()


def test_deduped_counter_emitted_for_waiter(monkeypatch):
    """
    inflight_guard.deduped must be incremented exactly once for every waiter
    thread that coalesces instead of executing its own query.

    Debugging use-case
    ------------------
    Comparing inflight_guard.deduped against the raw request count tells
    operators the dedupe hit rate.  A high rate is *expected* on a busy
    dashboard.  The deduped counter alone does not distinguish "saving work"
    from "adding wait time" — combine it with wait_ms for that.
    """
    fake_stats = _make_fake_stats_logger()
    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda key, value=None: (
            fake_stats.incr(key) if value is None else fake_stats.timing(key, value)
        ),
    )

    key = "obs-deduped-key"
    thread1_inside = threading.Event()
    thread2_done = threading.Event()

    def thread1_fn() -> None:
        with inflight_guard(key):
            thread1_inside.set()
            thread2_done.wait(timeout=5)

    def thread2_fn() -> None:
        thread1_inside.wait(timeout=5)
        with inflight_guard(key):
            pass
        thread2_done.set()

    t1 = threading.Thread(target=thread1_fn)
    t2 = threading.Thread(target=thread2_fn)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert "inflight_guard.deduped" in fake_stats.incr_calls, (
        "inflight_guard.deduped counter was not emitted for the coalesced waiter"
    )


def test_wait_ms_timing_emitted_for_waiter(monkeypatch):
    """
    inflight_guard.wait_ms must be emitted with a non-negative value for
    every waiter thread, regardless of whether the event fires normally or
    via a timeout.

    Debugging use-case
    ------------------
    * wait_ms ≪ query_duration  →  guard saved work (waiter barely blocked)
    * wait_ms ≈ query_duration  →  guard added latency (waiter waited as long
      as the query took; no net benefit vs executing independently)
    * wait_ms ≈ timeout         →  guard timed out; check timeout counter
    """
    fake_stats = _make_fake_stats_logger()
    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda key, value=None: (
            fake_stats.incr(key) if value is None else fake_stats.timing(key, value)
        ),
    )

    key = "obs-wait-ms-key"
    thread1_inside = threading.Event()
    thread2_done = threading.Event()

    def thread1_fn() -> None:
        with inflight_guard(key):
            thread1_inside.set()
            thread2_done.wait(timeout=5)

    def thread2_fn() -> None:
        thread1_inside.wait(timeout=5)
        with inflight_guard(key):
            pass
        thread2_done.set()

    t1 = threading.Thread(target=thread1_fn)
    t2 = threading.Thread(target=thread2_fn)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    timing_keys = [k for k, _ in fake_stats.timing_calls]
    assert "inflight_guard.wait_ms" in timing_keys, (
        "inflight_guard.wait_ms timing was not emitted for the coalesced waiter"
    )
    wait_ms_values = [v for k, v in fake_stats.timing_calls if k == "inflight_guard.wait_ms"]
    assert all(v >= 0 for v in wait_ms_values), (
        f"inflight_guard.wait_ms must be non-negative; got {wait_ms_values}"
    )


def test_timeout_counter_emitted_on_guard_timeout(monkeypatch):
    """
    inflight_guard.timeout must be incremented when event.wait() times out.

    Debugging use-case
    ------------------
    If timeout counter is high AND latency is not improving, the winner query
    is regularly exceeding QUERY_INFLIGHT_TIMEOUT_S.  In that scenario waiters
    block for the full timeout before a takeover occurs, so the guard is
    adding latency, not saving work.  Remedies: lower QUERY_INFLIGHT_TIMEOUT_S,
    optimise the slow query, or add a DB-side query timeout.
    """
    fake_stats = _make_fake_stats_logger()
    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda key, value=None: (
            fake_stats.incr(key) if value is None else fake_stats.timing(key, value)
        ),
    )

    key = "obs-timeout-key"
    original_timeout = query_cache_module._INFLIGHT_WAIT_TIMEOUT_S
    query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = 0.05

    thread1_inside = threading.Event()
    thread1_allowed_to_exit = threading.Event()

    def slow_thread() -> None:
        with inflight_guard(key):
            thread1_inside.set()
            thread1_allowed_to_exit.wait(timeout=10)

    def waiter_thread() -> None:
        thread1_inside.wait(timeout=5)
        with inflight_guard(key):
            pass

    try:
        t1 = threading.Thread(target=slow_thread)
        tw = threading.Thread(target=waiter_thread)
        t1.start()
        thread1_inside.wait(timeout=5)
        tw.start()
        tw.join(timeout=5)
        thread1_allowed_to_exit.set()
        t1.join(timeout=5)
    finally:
        query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = original_timeout

    assert "inflight_guard.timeout" in fake_stats.incr_calls, (
        "inflight_guard.timeout counter was not emitted when the guard wait timed out"
    )


def test_takeover_counter_emitted_on_takeover(monkeypatch):
    """
    inflight_guard.takeover must be incremented exactly once when a
    timeout-waiter wins the takeover race and becomes the new executor.

    Debugging use-case
    ------------------
    Each takeover represents one extra query execution beyond what was
    intended.  In steady state this should be near zero.  Spikes indicate
    queries routinely slower than the guard timeout — investigate with
    wait_ms and the underlying query duration metrics.
    """
    fake_stats = _make_fake_stats_logger()
    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda key, value=None: (
            fake_stats.incr(key) if value is None else fake_stats.timing(key, value)
        ),
    )

    key = "obs-takeover-key"
    original_timeout = query_cache_module._INFLIGHT_WAIT_TIMEOUT_S
    query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = 0.05

    thread1_inside = threading.Event()
    thread1_allowed_to_exit = threading.Event()

    def slow_thread() -> None:
        with inflight_guard(key):
            thread1_inside.set()
            thread1_allowed_to_exit.wait(timeout=10)

    def waiter_thread() -> None:
        thread1_inside.wait(timeout=5)
        with inflight_guard(key):
            pass

    try:
        t1 = threading.Thread(target=slow_thread)
        tw = threading.Thread(target=waiter_thread)
        t1.start()
        thread1_inside.wait(timeout=5)
        tw.start()
        tw.join(timeout=5)
        thread1_allowed_to_exit.set()
        t1.join(timeout=5)
    finally:
        query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = original_timeout

    assert "inflight_guard.takeover" in fake_stats.incr_calls, (
        "inflight_guard.takeover counter was not emitted for the takeover executor"
    )


def test_first_thread_emits_no_waiter_metrics(monkeypatch):
    """
    The first (winner) thread must NOT emit deduped, wait_ms, timeout, or
    takeover metrics — those are waiter-only signals.
    """
    fake_stats = _make_fake_stats_logger()
    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda key, value=None: (
            fake_stats.incr(key) if value is None else fake_stats.timing(key, value)
        ),
    )

    with inflight_guard("obs-first-only-key"):
        pass

    waiter_metrics = {
        "inflight_guard.deduped",
        "inflight_guard.wait_ms",
        "inflight_guard.timeout",
        "inflight_guard.takeover",
    }
    emitted = set(fake_stats.incr_calls) | {k for k, _ in fake_stats.timing_calls}
    unexpected = waiter_metrics & emitted
    assert not unexpected, (
        f"Winner thread emitted waiter-only metrics: {unexpected}"
    )


def test_lock_wait_ms_timing_emitted_for_every_thread(monkeypatch):
    """
    inflight_guard.lock_wait_ms must be emitted for EVERY thread (first and
    waiter alike) so operators can detect when _inflight_lock is becoming a
    serialisation bottleneck under high concurrency.

    Debugging use-case
    ------------------
    Under low load lock_wait_ms is near-zero (uncontended).  Sustained values
    above ~1 ms mean many threads are competing simultaneously for the same
    process-level lock, which is a signal to investigate dashboard fan-out or
    consider sharding the inflight table by key prefix.
    """
    fake_stats = _make_fake_stats_logger()
    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda key, value=None: (
            fake_stats.incr(key) if value is None else fake_stats.timing(key, value)
        ),
    )

    key = "obs-lock-wait-key"
    thread1_inside = threading.Event()
    thread2_done = threading.Event()

    def thread1_fn() -> None:
        with inflight_guard(key):
            thread1_inside.set()
            thread2_done.wait(timeout=5)

    def thread2_fn() -> None:
        thread1_inside.wait(timeout=5)
        with inflight_guard(key):
            pass
        thread2_done.set()

    t1 = threading.Thread(target=thread1_fn)
    t2 = threading.Thread(target=thread2_fn)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    timing_keys = [k for k, _ in fake_stats.timing_calls]
    lock_wait_count = timing_keys.count("inflight_guard.lock_wait_ms")
    # Both the winner (thread1) and the waiter (thread2) must emit this metric.
    assert lock_wait_count >= 2, (
        f"Expected lock_wait_ms from at least 2 threads, got {lock_wait_count}. "
        f"Timing calls: {fake_stats.timing_calls}"
    )
    lock_wait_values = [v for k, v in fake_stats.timing_calls if k == "inflight_guard.lock_wait_ms"]
    assert all(v >= 0 for v in lock_wait_values), (
        f"inflight_guard.lock_wait_ms must be non-negative; got {lock_wait_values}"
    )


def test_winner_query_ms_emitted_for_normal_winner(monkeypatch):
    """
    inflight_guard.winner_query_ms must be emitted by the first (winning)
    thread with a non-negative value so operators can compare it against
    wait_ms.

    Diagnostic use-case
    -------------------
    If dedupe hit rate is high but latency is not improving:
    * wait_ms ≈ winner_query_ms → guard is correct; the query itself is slow.
      Optimise the SQL or add a DB-side query timeout.
    * wait_ms ≫ winner_query_ms → waiters blocked longer than the query took;
      check lock_wait_ms or OS scheduling pressure.
    """
    fake_stats = _make_fake_stats_logger()
    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda key, value=None: (
            fake_stats.incr(key) if value is None else fake_stats.timing(key, value)
        ),
    )

    with inflight_guard("obs-winner-query-ms-key"):
        time.sleep(0.01)  # simulate a minimal query duration

    timing_keys = [k for k, _ in fake_stats.timing_calls]
    assert "inflight_guard.winner_query_ms" in timing_keys, (
        "inflight_guard.winner_query_ms was not emitted by the winning thread"
    )
    winner_ms_values = [
        v for k, v in fake_stats.timing_calls if k == "inflight_guard.winner_query_ms"
    ]
    assert all(v >= 0 for v in winner_ms_values), (
        f"inflight_guard.winner_query_ms must be non-negative; got {winner_ms_values}"
    )


def test_winner_query_ms_emitted_for_takeover_thread(monkeypatch):
    """
    inflight_guard.winner_query_ms must also be emitted by the takeover thread
    so that both normal-winner and takeover executions are observable.

    Diagnostic use-case
    -------------------
    Each takeover = one extra query execution.  Its duration is emitted as
    winner_query_ms alongside the normal winner's duration, allowing operators
    to see how long the replacement query took vs the original wait time.
    """
    fake_stats = _make_fake_stats_logger()
    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda key, value=None: (
            fake_stats.incr(key) if value is None else fake_stats.timing(key, value)
        ),
    )

    key = "obs-takeover-winner-query-ms-key"
    original_timeout = query_cache_module._INFLIGHT_WAIT_TIMEOUT_S
    query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = 0.05

    thread1_inside = threading.Event()
    thread1_allowed_to_exit = threading.Event()

    def slow_thread() -> None:
        with inflight_guard(key):
            thread1_inside.set()
            thread1_allowed_to_exit.wait(timeout=10)

    def waiter_thread() -> None:
        thread1_inside.wait(timeout=5)
        with inflight_guard(key):
            pass  # takeover winner does minimal work

    try:
        t1 = threading.Thread(target=slow_thread)
        tw = threading.Thread(target=waiter_thread)
        t1.start()
        thread1_inside.wait(timeout=5)
        tw.start()
        tw.join(timeout=5)
        thread1_allowed_to_exit.set()
        t1.join(timeout=5)
    finally:
        query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = original_timeout

    timing_keys = [k for k, _ in fake_stats.timing_calls]
    assert "inflight_guard.winner_query_ms" in timing_keys, (
        "inflight_guard.winner_query_ms was not emitted for the takeover thread"
    )


def test_winner_query_ms_not_emitted_by_non_executing_waiter(monkeypatch):
    """
    A waiter that is released normally (no timeout, no takeover) must NOT
    emit winner_query_ms — it did not execute a query.

    This ensures operators can use winner_query_ms as a clean signal of actual
    query executions without noise from threads that only read from cache.
    """
    fake_stats = _make_fake_stats_logger()
    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda key, value=None: (
            fake_stats.incr(key) if value is None else fake_stats.timing(key, value)
        ),
    )

    key = "obs-waiter-no-winner-ms-key"
    thread1_inside = threading.Event()
    thread2_done = threading.Event()
    waiter_timing_calls: list[tuple[str, float]] = []

    def thread1_fn() -> None:
        with inflight_guard(key):
            thread1_inside.set()
            thread2_done.wait(timeout=5)

    def thread2_fn() -> None:
        # Capture timing calls emitted ONLY while thread2 is running.
        thread1_inside.wait(timeout=5)
        before = len(fake_stats.timing_calls)
        with inflight_guard(key):
            pass  # is_first=False; should NOT emit winner_query_ms
        waiter_timing_calls.extend(fake_stats.timing_calls[before:])
        thread2_done.set()

    t1 = threading.Thread(target=thread1_fn)
    t2 = threading.Thread(target=thread2_fn)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    waiter_emitted_keys = [k for k, _ in waiter_timing_calls]
    assert "inflight_guard.winner_query_ms" not in waiter_emitted_keys, (
        "A non-executing waiter thread emitted winner_query_ms, which would "
        f"skew execution-duration metrics.  Calls from waiter: {waiter_timing_calls}"
    )


# ---------------------------------------------------------------------------
# E2E Concurrency Simulation
#
# These tests simulate a real dashboard under load: N charts sharing the same
# datasource+filters all POST to /api/v1/chart/data at the same moment.
# They verify three properties that must hold in production:
#
#   1. Duplicate query reduction — only ONE DB query executes regardless of N.
#   2. No latency regression — N concurrent requests complete in roughly the
#      same wall time as a single request, not N× longer.
#   3. Fallback correctness — winner failure (sentinel) and guard timeout
#      (takeover) both resolve cleanly without extra query executions.
#
# Each test uses a threading.Barrier so all threads fire simultaneously,
# reproducing the burst that occurs when a browser loads a dashboard.
# A fake "cache" dict replaces the real cache backend; a fake "DB query"
# is a time.sleep() call whose invocation count is tracked.
# ---------------------------------------------------------------------------


def _run_concurrent_chart_requests(
    n_charts: int,
    key: str,
    query_fn,  # callable(); called by the "winner" thread only
    cache: dict,
    errors: list,
) -> dict:
    """
    Launch *n_charts* threads that all arrive at the same cache miss at once
    (via a Barrier) and simulate the get_df_payload critical section.

    *query_fn* is called once by the winning thread to populate *cache*.
    Returns a dict with ``exec_count`` and ``wall_time_s``.
    """
    exec_count = 0
    exec_lock = threading.Lock()
    barrier = threading.Barrier(n_charts)
    start_time: list[float] = []
    end_events: list[threading.Event] = [threading.Event() for _ in range(n_charts)]

    def chart_request(idx: int) -> None:
        nonlocal exec_count
        try:
            barrier.wait(timeout=10)
            if idx == 0:
                start_time.append(time.monotonic())

            cache_hit = key in cache
            if not cache_hit:
                with inflight_guard(key) as is_first:
                    if not is_first:
                        cache_hit = key in cache
                    if is_first or not cache_hit:
                        with exec_lock:
                            exec_count += 1
                        result = query_fn()
                        cache[key] = result
        except (RuntimeError, threading.BrokenBarrierError) as exc:
            errors.append(f"thread-{idx}: {exc}")
        finally:
            end_events[idx].set()

    threads = [
        threading.Thread(target=chart_request, args=(i,), name=f"chart-{i}")
        for i in range(n_charts)
    ]
    for t in threads:
        t.start()
    for ev in end_events:
        ev.wait(timeout=30)
    for t in threads:
        t.join(timeout=5)

    wall = time.monotonic() - start_time[0] if start_time else 0.0
    return {"exec_count": exec_count, "wall_time_s": wall}


def test_high_concurrency_duplicate_queries_reduced():
    """
    Simulates 15 dashboard charts all hitting the same cache miss simultaneously.

    Expected: exactly 1 DB query executes (exec_count == 1).

    This is the primary proof that inflight_guard eliminates the thundering
    herd: without the guard, all 15 charts would each execute the same SQL.
    With the guard, 14 charts coalesce behind the first and read its result
    from cache.

    E2E mapping:
    - n_charts=15 → 15 concurrent POST /api/v1/chart/data for the same query
    - fake_query() → the SQL query sent to the database
    - cache dict → the cache backend (Redis / in-process)
    """
    n_charts = 15
    key = "e2e-high-concurrency-key"
    cache: dict = {}
    errors: list[str] = []
    query_call_count = 0
    count_lock = threading.Lock()

    def fake_query():
        nonlocal query_call_count
        with count_lock:
            query_call_count += 1
        # Simulate a realistic query taking 30 ms.
        time.sleep(0.03)
        return "result-data"

    result = _run_concurrent_chart_requests(n_charts, key, fake_query, cache, errors)

    if errors:
        pytest.fail(f"Thread errors during concurrent simulation: {errors}")

    assert result["exec_count"] == 1, (
        f"Expected exactly 1 DB query execution for {n_charts} concurrent chart "
        f"requests (inflight_guard should coalesce), but {result['exec_count']} "
        f"executions occurred. This means the thundering herd was NOT prevented."
    )
    assert cache.get(key) == "result-data", (
        "Cache was not populated after the single query execution"
    )
    assert query_call_count == 1, (
        f"fake_query() was called {query_call_count} times; expected exactly 1"
    )


def test_no_latency_regression_concurrent_vs_sequential():
    """
    Verifies that N concurrent chart requests complete in roughly the same wall
    time as a single sequential request, not in N× longer.

    When the guard is working correctly, all N threads are gated behind the
    single winner query.  The total wall time should be ≈ query_duration, not
    N × query_duration.

    Latency regression definition used here:
        wall_time > 3.5 × query_duration → regression detected.

    The generous factor accommodates:
    - Thread scheduling jitter (all threads start from a Barrier, not truly
      simultaneously from the OS scheduler's perspective)
    - Barrier wait itself adds a small amount
    - CI runner variability

    E2E mapping:
    - 8 simultaneous chart renders on a dashboard
    - 40 ms DB query (simulated)
    - Expectation: dashboard loads in ~40–100 ms total, not 320 ms (8×40)
    """
    n_charts = 8
    key = "e2e-latency-regression-key"
    query_duration_s = 0.04  # 40 ms simulated DB query
    cache: dict = {}
    errors: list[str] = []

    def timed_query():
        time.sleep(query_duration_s)
        return "latency-result"

    result = _run_concurrent_chart_requests(
        n_charts, key, timed_query, cache, errors
    )

    if errors:
        pytest.fail(f"Thread errors: {errors}")

    assert result["exec_count"] == 1, (
        f"exec_count={result['exec_count']}; expected 1 (deduplication must work "
        f"before we can reason about latency)"
    )

    max_allowed_s = query_duration_s * 3.5
    assert result["wall_time_s"] <= max_allowed_s, (
        f"Latency regression detected: {n_charts} concurrent requests took "
        f"{result['wall_time_s']:.3f}s, but the single query only takes "
        f"{query_duration_s:.3f}s. Max allowed is {max_allowed_s:.3f}s "
        f"({n_charts}× regression would be {n_charts * query_duration_s:.3f}s). "
        f"The guard appears to be adding wait time rather than saving work."
    )


def test_winner_failure_sentinel_skips_waiter_reexecution():
    """
    Simulates the fallback path where the winning thread fails mid-query.

    Expected behaviour:
    1. Winner thread raises an exception inside the guard.
    2. Before raising, it writes an error sentinel to the fake cache.
       (In production, QueryCacheManager.set_error_sentinel() does this.)
    3. inflight_guard's finally block fires the Event, unblocking waiters.
    4. Waiters re-read the cache, find the sentinel, and skip re-execution.

    Result:
    - exec_count == 1  (winner attempted; no waiter re-executed)
    - No thread sees is_first=True a second time (no thundering herd on failure)

    E2E mapping:
    - Winner hits a DB connection error or query timeout.
    - Sentinel prevents all N-1 waiting charts from issuing the same failing query.
    - Every chart gets the same error response from the sentinel, avoiding N
      concurrent identical failures.
    """
    n_charts = 6
    key = "e2e-winner-failure-key"
    # Shared fake cache.  Sentinel format matches QueryCacheManager convention.
    cache: dict = {}
    exec_count = 0
    exec_lock = threading.Lock()
    barrier = threading.Barrier(n_charts)
    errors: list[str] = []

    def chart_request_with_failure() -> None:
        nonlocal exec_count
        try:
            barrier.wait(timeout=10)

            cache_hit = key in cache
            if not cache_hit:
                with inflight_guard(key) as is_first:
                    if not is_first:
                        # Re-read after unblocking.
                        cache_hit = key in cache

                    if is_first or not cache_hit:
                        with exec_lock:
                            exec_count += 1

                        if is_first:
                            # Simulate winner failing: write sentinel, then raise.
                            # In production this is QueryCacheManager.set_error_sentinel().
                            cache[key] = {"__error__": "simulated DB error"}
                            raise RuntimeError("simulated DB error")
                        # Waiters: if sentinel is present, skip execution.
                        # (is_first=False AND cache_hit=True means sentinel found)
        except RuntimeError:
            pass  # Expected for the winner thread only.
        except threading.BrokenBarrierError as exc:
            errors.append(str(exc))

    threads = [
        threading.Thread(target=chart_request_with_failure, name=f"chart-{i}")
        for i in range(n_charts)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    if errors:
        pytest.fail(f"Unexpected thread errors: {errors}")

    assert exec_count == 1, (
        f"Expected exactly 1 execution attempt (winner only) after winner failure. "
        f"Waiters should read the error sentinel and skip re-execution. "
        f"Got exec_count={exec_count}. "
        f"This means the sentinel-on-failure path is not preventing waiter "
        f"re-execution — every failed chart is issuing the same failing query."
    )
    assert "__error__" in cache.get(key, {}), (
        "Error sentinel must be written to cache by the failing winner so that "
        "waiters can detect the failure without re-executing."
    )


def test_timeout_fallback_caps_extra_executions_at_one():
    """
    Simulates the timeout + takeover fallback path under realistic concurrency.

    Scenario:
    - Thread-1 (winner) starts a very slow query (exceeds guard timeout).
    - N-1 waiter threads all time out simultaneously.
    - Exactly ONE waiter thread wins the takeover race and executes the query.
    - All other waiters block until the takeover completes and then read cache.

    Expected:
    - exec_count == 2  (original winner attempt + 1 takeover)
    - NOT exec_count == N (thundering herd on timeout)

    E2E mapping:
    - A slow query (e.g. a heavy aggregation) exceeds QUERY_INFLIGHT_TIMEOUT_S.
    - Without the takeover mechanism: all N charts would retry simultaneously,
      multiplying DB load N×.
    - With the takeover mechanism: only 1 additional query fires; all others
      wait for it.

    Note: exec_count may be 1 if Thread-1 completes before the takeover
    thread reaches the cache-write step (timing-dependent), but it must never
    exceed 2.
    """
    n_charts = 8
    key = "e2e-timeout-takeover-key"
    cache: dict = {}
    exec_count = 0
    exec_lock = threading.Lock()
    errors: list[str] = []

    original_timeout = query_cache_module._INFLIGHT_WAIT_TIMEOUT_S
    query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = 0.05  # 50 ms timeout

    thread1_inside = threading.Event()
    thread1_allowed_to_exit = threading.Event()

    try:

        def slow_winner() -> None:
            nonlocal exec_count
            with inflight_guard(key) as is_first:
                if is_first:
                    with exec_lock:
                        exec_count += 1
                    thread1_inside.set()
                    # Hold the slot past the timeout so all waiters time out.
                    thread1_allowed_to_exit.wait(timeout=10)
                    cache[key] = "result-from-original-winner"

        def chart_waiter() -> None:
            nonlocal exec_count
            try:
                thread1_inside.wait(timeout=5)
                with inflight_guard(key) as is_first:
                    if not is_first:
                        # Re-read cache after unblocking.
                        pass
                    if is_first or key not in cache:
                        with exec_lock:
                            exec_count += 1
                        cache[key] = "result-from-takeover"
            except threading.BrokenBarrierError as exc:
                errors.append(str(exc))

        t1 = threading.Thread(target=slow_winner, name="chart-winner")
        waiters = [
            threading.Thread(target=chart_waiter, name=f"chart-waiter-{i}")
            for i in range(n_charts - 1)
        ]

        t1.start()
        thread1_inside.wait(timeout=5)
        for w in waiters:
            w.start()

        # Let waiters time out and resolve (via takeover).
        for w in waiters:
            w.join(timeout=10)

        # Allow the original winner to finish.
        thread1_allowed_to_exit.set()
        t1.join(timeout=5)

    finally:
        query_cache_module._INFLIGHT_WAIT_TIMEOUT_S = original_timeout

    if errors:
        pytest.fail(f"Thread errors: {errors}")

    assert exec_count <= 2, (
        f"Timeout fallback allowed {exec_count} query executions across {n_charts} "
        f"concurrent chart requests. Expected ≤ 2 (original winner + at most 1 "
        f"takeover). This means the takeover mechanism did not prevent the thundering "
        f"herd on timeout — {exec_count - 2} extra executions occurred."
    )
    assert exec_count >= 1, (
        "At least 1 execution must have occurred (the original winner)"
    )


def test_metrics_deduped_count_matches_waiter_count(monkeypatch):
    """
    Verifies that inflight_guard.deduped is incremented exactly once per waiter
    thread — no more, no less — across a high-concurrency simulation.

    This is the key debugging signal for the question "is dedupe hit rate high?"
    A high deduped count combined with exec_count==1 confirms that deduplication
    is working.  If deduped < N-1, some waiters bypassed the guard.  If
    deduped > N-1, the counter is being double-incremented.

    E2E mapping:
    - N charts share the same query key → N-1 should be deduped.
    - deduped / (deduped + 1) ≈ dedupe hit rate percentage.
    - If hit rate is high (N-1 deduped) but latency is still bad:
      compare wait_ms vs winner_query_ms to identify whether the query
      itself is slow or the guard is adding overhead.
    """
    n_charts = 10
    key = "e2e-deduped-count-key"
    cache: dict = {}
    errors: list[str] = []
    fake_stats = _make_fake_stats_logger()

    monkeypatch.setattr(
        query_cache_module,
        "_emit_stat",
        lambda stat_key, value=None: (
            fake_stats.incr(stat_key)
            if value is None
            else fake_stats.timing(stat_key, value)
        ),
    )

    def fake_query():
        time.sleep(0.02)
        return "metric-result"

    result = _run_concurrent_chart_requests(n_charts, key, fake_query, cache, errors)

    if errors:
        pytest.fail(f"Thread errors: {errors}")

    assert result["exec_count"] == 1, (
        f"Deduplication must work (exec_count==1) before the deduped metric "
        f"can be trusted. Got exec_count={result['exec_count']}."
    )

    deduped_count = fake_stats.incr_calls.count("inflight_guard.deduped")
    expected_deduped = n_charts - 1  # Every thread except the winner is a waiter.
    assert deduped_count == expected_deduped, (
        f"inflight_guard.deduped was incremented {deduped_count} times, "
        f"expected {expected_deduped} (one per waiter thread). "
        f"Dedupe hit rate = {deduped_count}/{n_charts} "
        f"({100 * deduped_count / n_charts:.0f}%). "
        f"If this is lower than expected, some waiters bypassed the guard. "
        f"All incr_calls: {fake_stats.incr_calls}"
    )

    # Sanity: winner_query_ms should have been emitted exactly once (by the winner).
    winner_ms_count = sum(
        1 for k, _ in fake_stats.timing_calls if k == "inflight_guard.winner_query_ms"
    )
    assert winner_ms_count == 1, (
        f"inflight_guard.winner_query_ms should be emitted once (by the winner), "
        f"got {winner_ms_count}. This metric is needed to compare against wait_ms "
        f"to determine if the guard is saving work or adding latency."
    )
