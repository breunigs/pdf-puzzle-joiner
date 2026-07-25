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
