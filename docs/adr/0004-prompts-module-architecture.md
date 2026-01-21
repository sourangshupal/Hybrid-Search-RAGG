# ADR-0004: Prompts Module Architecture

## Status
Accepted

## Context
System prompts scattered across codebase, hard to maintain and customize.

## Decision
Centralize all prompts in `src/hybridrag/prompts/` module with factory functions.

## Consequences
**Positive**: Easy customization, clear organization
**Negative**: Additional module to maintain

## Date
2026-01-20
