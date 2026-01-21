# ADR-0005: Dual Filter Builder Systems

## Status
Accepted

## Context
Vector search and Atlas Search use different filter syntaxes. Easy to mix them up.

## Decision
Create separate type-safe builders for each: `build_vector_search_filters()` and `build_atlas_search_filters()`.

## Consequences
**Positive**: Type safety, clear separation, prevents syntax errors
**Negative**: Two similar APIs to learn

## Date
2026-01-20
