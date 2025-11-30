"""Unit tests for state management."""

from docmana.state import override, merge_dicts


class TestStateReducers:
    """Test state reducer functions."""

    def test_override_replaces_value(self):
        """Override should always return the new value."""
        assert override("old", "new") == "new"
        assert override(123, 456) == 456

    def test_override_with_none(self):
        """Override should handle None values."""
        assert override("old", None) is None
        assert override(None, "new") == "new"

    def test_override_with_complex_types(self):
        """Override should work with lists and dicts."""
        old_list = [1, 2, 3]
        new_list = [4, 5]
        assert override(old_list, new_list) == [4, 5]

        old_dict = {"a": 1}
        new_dict = {"b": 2}
        assert override(old_dict, new_dict) == {"b": 2}

    def test_merge_dicts_empty(self):
        """Merging empty dicts should return empty dict."""
        assert merge_dicts({}, {}) == {}

    def test_merge_dicts_adds_new_keys(self):
        """New keys should be added to result."""
        old = {"a": "1"}
        new = {"b": "2"}
        result = merge_dicts(old, new)
        assert result == {"a": "1", "b": "2"}

    def test_merge_dicts_overwrites_existing(self):
        """Existing keys should be overwritten."""
        old = {"a": "1", "b": "2"}
        new = {"b": "3"}
        result = merge_dicts(old, new)
        assert result == {"a": "1", "b": "3"}

    def test_merge_dicts_with_none_new(self):
        """Merging with None should return old dict."""
        old = {"a": "1"}
        result = merge_dicts(old, None)
        assert result == {"a": "1"}
