import pytest
from unittest.mock import MagicMock
from src.task_def_v4 import TaskManualMatcher

class TestTaskMatching:
    @pytest.fixture
    def matcher(self):
        # Setup mock document processor with sample data
        mock_dp = MagicMock()
        mock_dp.manual_full_texts = {
            'man1': 'sensor calibration steps',
            'man2': 'lubrication procedure for bearings'
        }
        return TaskManualMatcher(mock_dp)

    def test_positive_task_match(self, matcher):
        """Test successful regex pattern match"""
        # Test exact match in text
        result = matcher._task_match(
            text="sensor calibration steps",
            task="calibration"
        )
        assert result is True

    def test_fuzzy_match_fallback(self, matcher):
        """Test fuzzy matching when exact match fails"""
        # Test partial match with typo
        result = matcher._task_match(
            text="calibratiun steps",  # Intentional typo
            task="calibration"
        )
        assert result is True

    def test_no_match_scenario(self, matcher):
        """Test non-matching task and text"""
        result = matcher._task_match(
            text="bearing maintenance",
            task="calibration"
        )
        assert result is False

    def test_regex_caching(self, matcher):
        """Test regex pattern caching mechanism"""
        # First call should create cache entry
        matcher._task_match("text", "test task")
        assert "test task" in matcher.regex_cache

        # Second call should use cached regex
        cache_size = len(matcher.regex_cache)
        matcher._task_match("text", "test task")
        assert len(matcher.regex_cache) == cache_size