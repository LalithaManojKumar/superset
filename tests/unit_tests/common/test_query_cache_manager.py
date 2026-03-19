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
