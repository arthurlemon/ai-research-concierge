"""Unit tests for research tools."""

import pytest
from unittest.mock import AsyncMock, patch

from docmana.tools import _match_topic, knowledge_lookup, tavily_search


class TestKnowledgeLookup:
    """Test local knowledge base lookup."""

    def test_match_topic_exact_match(self):
        """Should match when all tokens are present."""
        result = _match_topic("python vs javascript backend")
        assert result is not None
        assert "Performance" in result

    def test_match_topic_partial_match(self):
        """Should fallback to partial match."""
        result = _match_topic("python backend")
        assert result is not None

    def test_match_topic_case_insensitive(self):
        """Matching should be case-insensitive."""
        result = _match_topic("PYTHON VS JAVASCRIPT")
        assert result is not None

    def test_match_topic_no_match(self):
        """Should return None for no match."""
        result = _match_topic("quantum computing blockchain")
        assert result is None

    @pytest.mark.asyncio
    async def test_knowledge_lookup_success(self):
        """Should return KB content for valid topic."""
        result = await knowledge_lookup.ainvoke({"topic": "python backend"})
        assert "Performance" in result or "Python" in result

    @pytest.mark.asyncio
    async def test_knowledge_lookup_no_match(self):
        """Should return message when no match found."""
        result = await knowledge_lookup.ainvoke({"topic": "nonexistent topic xyz"})
        assert "Aucune donnée locale trouvée" in result


class TestTavilySearch:
    """Test Tavily web search tool."""

    @pytest.mark.asyncio
    async def test_tavily_search_success(self):
        """Should format results correctly on success."""
        mock_response = {
            "results": [
                {
                    "title": "Test Article",
                    "url": "https://example.com",
                    "content": "Test content here",
                }
            ]
        }

        # Mock at the import level inside the function
        with patch("tavily.AsyncTavilyClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_instance.search.return_value = mock_response
            mock_client_class.return_value = mock_instance

            result = await tavily_search.ainvoke({"query": "test query"})

            assert "Test Article" in result
            assert "https://example.com" in result
            assert "Test content" in result

    @pytest.mark.asyncio
    async def test_tavily_search_missing_key(self, monkeypatch):
        """Should return error when API key missing."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        result = await tavily_search.ainvoke({"query": "test"})
        assert "TAVILY_API_KEY non trouvée" in result

    @pytest.mark.asyncio
    async def test_tavily_search_exception(self):
        """Should handle API exceptions gracefully."""
        with patch("tavily.AsyncTavilyClient") as mock_client_class:
            mock_instance = AsyncMock()
            mock_instance.search.side_effect = Exception("API Error")
            mock_client_class.return_value = mock_instance

            result = await tavily_search.ainvoke({"query": "test"})
            assert "Erreur lors de la recherche Tavily" in result
