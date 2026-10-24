from puzzle_joiner.priority import _build_low_prio_prefix, LOW_PRIO


class TestPriority:
    def test_prefix_is_list(self):
        assert isinstance(LOW_PRIO, list)

    def test_prefix_contains_strings(self):
        for item in LOW_PRIO:
            assert isinstance(item, str)

    def test_build_returns_list(self):
        result = _build_low_prio_prefix()
        assert isinstance(result, list)


class TestMainThreadIsSpared:
    """Background work must not idle-schedule the thread drawing the UI.

    `_restore_process_priority` cannot undo it — raising a nice value back and
    leaving SCHED_IDLE both need CAP_SYS_NICE — so one background task would
    otherwise leave the whole app sluggish for the rest of the session.
    """

    def test_the_main_thread_is_not_visited(self):
        import threading
        from puzzle_joiner.priority import _for_all_threads

        started = threading.Event()
        release = threading.Event()

        def park():
            started.set()
            release.wait(5)

        helper = threading.Thread(target=park, daemon=True)
        helper.start()
        started.wait(5)
        try:
            seen = []
            _for_all_threads(seen.append)
            assert seen, "no background threads were visited at all"
            assert threading.main_thread().native_id not in seen
            assert helper.native_id in seen
        finally:
            release.set()
            helper.join(5)
