"""Tests for lexical prefilter builder."""

from hybridrag.enhancements.filters.lexical_prefilters import (
    LexicalPrefilterConfig,
    build_lexical_prefilters,
)


class TestLexicalPrefilterConfig:
    """Test lexical prefilter configuration."""

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        config = LexicalPrefilterConfig()
        assert config.text_filters == []
        assert config.fuzzy_filters == []
        assert config.phrase_filters == []
        assert config.wildcard_filters == []
        assert config.range_filters == {}
        assert config.geo_filters == []
        assert config.query_string_filter is None


class TestBuildLexicalPrefilters:
    """Test lexical prefilter builder for $search.vectorSearch."""

    def test_empty_config_returns_empty_dict(self):
        """Empty config should return empty filter dict."""
        config = LexicalPrefilterConfig()
        result = build_lexical_prefilters(config)
        assert result == {}

    def test_text_filter(self):
        """Text filter uses Atlas 'text' operator."""
        config = LexicalPrefilterConfig(
            text_filters=[{"path": "content", "query": "mongodb"}]
        )
        result = build_lexical_prefilters(config)

        assert "compound" in result
        assert "filter" in result["compound"]
        text_filter = result["compound"]["filter"][0]
        assert text_filter["text"]["query"] == "mongodb"
        assert text_filter["text"]["path"] == "content"
