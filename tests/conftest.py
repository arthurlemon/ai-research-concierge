"""Shared fixtures and configuration for tests."""

import pytest
from unittest.mock import AsyncMock

from docmana.state import GraphState

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def sample_state():
    """Base GraphState for testing."""
    return GraphState(
        messages=[],
        user_query="Test query",
        sub_questions=[],
        tool_results={},
        final_answer=None,
        error=None,
    )


@pytest.fixture
def state_with_sub_questions():
    """State after successful query analysis."""
    return GraphState(
        messages=[],
        user_query="Complex question about Python",
        sub_questions=["What is Python?", "Why use Python?"],
        tool_results={},
        final_answer=None,
        error=None,
    )


@pytest.fixture
def state_with_evidence():
    """State after successful evidence gathering."""
    return GraphState(
        messages=[],
        user_query="Complex question about Python",
        sub_questions=["What is Python?", "Why use Python?"],
        tool_results={
            "What is Python?": "Python is a high-level programming language.",
            "Why use Python?": "Python is easy to learn and has great libraries.",
        },
        final_answer=None,
        error=None,
    )


@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""

    def _create(content: str):
        response = AsyncMock()
        response.content = content
        return response

    return _create


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Auto-mock API keys for all tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
