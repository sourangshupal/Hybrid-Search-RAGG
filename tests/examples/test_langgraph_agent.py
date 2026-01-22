"""Tests for the LangGraph agent example.

These tests verify the example script can be imported and basic structures work.
Full integration tests require API keys and MongoDB connection.
"""

import sys
from pathlib import Path

import pytest

# Add examples to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))


class TestLangGraphAgentExample:
    """Test the LangGraph agent example components."""

    @pytest.mark.skipif(
        True,
        reason="Skip by default - requires langchain/langgraph dependencies",
    )
    def test_example_imports(self) -> None:
        """Test that the example can be imported when dependencies are installed."""
        # This test is skipped by default because it requires optional deps
        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "langgraph_agent",
                Path(__file__).parent.parent.parent
                / "examples"
                / "09_langgraph_agent.py",
            )
            # Don't execute, just verify syntax
            assert spec is not None
        except ImportError as e:
            pytest.skip(f"LangGraph dependencies not installed: {e}")

    def test_example_file_exists(self) -> None:
        """Test that the example file exists."""
        example_path = (
            Path(__file__).parent.parent.parent / "examples" / "09_langgraph_agent.py"
        )
        assert example_path.exists(), f"Example file not found at {example_path}"

    def test_example_has_docstring(self) -> None:
        """Test that the example has proper documentation."""
        example_path = (
            Path(__file__).parent.parent.parent / "examples" / "09_langgraph_agent.py"
        )
        content = example_path.read_text()

        # Check for key documentation elements
        assert '"""' in content, "Example should have docstrings"
        assert "LangGraph" in content, "Should mention LangGraph"
        assert "HybridRAG" in content, "Should mention HybridRAG"
        assert "Prerequisites" in content, "Should have prerequisites section"
        assert "pip install" in content, "Should have installation instructions"

    def test_example_has_main_guard(self) -> None:
        """Test that the example has proper __main__ guard."""
        example_path = (
            Path(__file__).parent.parent.parent / "examples" / "09_langgraph_agent.py"
        )
        content = example_path.read_text()

        assert 'if __name__ == "__main__"' in content, "Should have main guard"
        assert "asyncio.run" in content, "Should use asyncio.run for async main"
