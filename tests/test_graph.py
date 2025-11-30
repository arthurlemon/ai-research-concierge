"""Unit tests for graph nodes and routing."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from docmana.state import GraphState
from docmana.graph import (
    _after_analyze,
    _after_gather,
    analyze_query,
    gather_evidence,
    AnalysisPlan,
)


class TestRoutingLogic:
    """Test conditional routing functions."""

    def test_after_analyze_success_route(self, state_with_sub_questions):
        """Should route to 'gather' when sub-questions exist."""
        result = _after_analyze(state_with_sub_questions)
        assert result == "gather"

    def test_after_analyze_error_route(self):
        """Should route to 'error' when error present."""
        state = GraphState(
            messages=[],
            user_query="test",
            sub_questions=[],
            tool_results={},
            final_answer=None,
            error="Analysis failed",
        )
        result = _after_analyze(state)
        assert result == "error"

    def test_after_analyze_empty_questions_route(self):
        """Should route to 'error' when no sub-questions."""
        state = GraphState(
            messages=[],
            user_query="test",
            sub_questions=[],
            tool_results={},
            final_answer=None,
            error=None,
        )
        result = _after_analyze(state)
        assert result == "error"

    def test_after_gather_success_route(self, state_with_evidence):
        """Should route to 'synthesize' when results exist."""
        result = _after_gather(state_with_evidence)
        assert result == "synthesize"

    def test_after_gather_error_route(self):
        """Should route to 'error' when error present."""
        state = GraphState(
            messages=[],
            user_query="test",
            sub_questions=["Q1"],
            tool_results={},
            final_answer=None,
            error="Gathering failed",
        )
        result = _after_gather(state)
        assert result == "error"

    def test_after_gather_empty_results_route(self):
        """Should route to 'error' when no results."""
        state = GraphState(
            messages=[],
            user_query="test",
            sub_questions=["Q1"],
            tool_results={},
            final_answer=None,
            error=None,
        )
        result = _after_gather(state)
        assert result == "error"


class TestAnalyzeQueryNode:
    """Test analyze_query graph node."""

    @pytest.mark.asyncio
    async def test_analyze_query_success(self, sample_state):
        """Should decompose query into sub-questions."""
        mock_plan = AnalysisPlan(sub_questions=["Question 1?", "Question 2?"])

        with patch("docmana.graph.configurable_model") as mock_model:
            # Create the final model that will be called with ainvoke
            mock_final_model = AsyncMock()
            mock_final_model.ainvoke = AsyncMock(return_value=mock_plan)

            # Mock with_config to return a model that has with_structured_output
            mock_with_config = Mock()
            mock_with_config.with_structured_output = Mock(
                return_value=mock_final_model
            )

            # Mock with_structured_output to return something with with_config
            mock_with_structured = Mock()
            mock_with_structured.with_config = Mock(return_value=mock_final_model)

            # Setup the chain: configurable_model.with_structured_output().with_config()
            mock_model.with_structured_output = Mock(return_value=mock_with_structured)

            result = await analyze_query(sample_state, {})

        assert "sub_questions" in result
        assert len(result["sub_questions"]) == 2
        assert result["sub_questions"][0] == "Question 1?"

    @pytest.mark.asyncio
    async def test_analyze_query_empty_query_error(self):
        """Should return error for empty query."""
        state = GraphState(
            messages=[],
            user_query="",
            sub_questions=[],
            tool_results={},
            final_answer=None,
            error=None,
        )

        result = await analyze_query(state, {})

        assert "error" in result
        assert "Aucune question" in result["error"]


class TestGatherEvidenceNode:
    """Test gather_evidence graph node."""

    @pytest.mark.asyncio
    async def test_gather_evidence_empty_questions_error(self, sample_state):
        """Should return error when no sub-questions."""
        result = await gather_evidence(sample_state, {})

        assert "error" in result
        assert "Aucune sous-question" in result["error"]

    @pytest.mark.asyncio
    async def test_gather_evidence_basic_flow(self, state_with_sub_questions):
        """Should gather evidence for sub-questions."""
        with (
            patch("docmana.graph.tavily_search") as mock_tavily,
            patch("docmana.graph.knowledge_lookup") as mock_kb,
            patch("docmana.graph.configurable_model") as mock_model,
        ):
            # Mock tool responses
            mock_tavily.ainvoke = AsyncMock(return_value="Tavily result data")
            mock_kb.ainvoke = AsyncMock(return_value="KB result data")

            # Mock LLM summarization
            mock_response = AsyncMock()
            mock_response.content = "Summarized evidence"
            mock_configured = AsyncMock()
            mock_configured.ainvoke = AsyncMock(return_value=mock_response)
            mock_model.with_config = Mock(return_value=mock_configured)

            result = await gather_evidence(state_with_sub_questions, {})

        assert "tool_results" in result
        # Should have results for at least some questions
        assert len(result["tool_results"]) > 0
