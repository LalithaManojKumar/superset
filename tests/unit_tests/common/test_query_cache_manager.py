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
