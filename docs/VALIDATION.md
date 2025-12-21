# Documentation Validation Report

This document validates that all documentation matches the actual codebase implementation.

## Validation Date
2025-01-XX

## Code References

### Core API Validation

#### Factory Function
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:1039-1063
async def create_hybridrag(
    settings: Settings | None = None,
    working_dir: str = "./hybridrag_workspace",
    auto_initialize: bool = True,
) -> HybridRAG:
```

✅ **DOCUMENTED CORRECTLY** in `docs/api.md`

#### HybridRAG Class Initialization
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:134-168
@dataclass
class HybridRAG:
    settings: Settings = field(default_factory=get_settings)
    working_dir: str = field(default="./hybridrag_workspace")
    _rag_engine: _RAGEngine | None = field(default=None, repr=False)
    _llm_func: Callable | None = field(default=None, repr=False)
    _memory: ConversationMemory | None = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    async def initialize(self) -> None:
```

✅ **DOCUMENTED CORRECTLY** - Must call `initialize()` or use `create_hybridrag()` with `auto_initialize=True`

#### Insert Method
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:333-388
async def insert(
    self,
    documents: str | Sequence[str],
    ids: Sequence[str] | None = None,
    file_paths: Sequence[str] | None = None,
) -> None:
```

✅ **DOCUMENTED CORRECTLY** in `docs/api.md` - Fixed from incorrect `ingest()` method

#### Ingest Files Method
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:390-538
async def ingest_files(
    self,
    folder_path: str | "Path",
    config: IngestionConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[IngestionResult]:
```

✅ **DOCUMENTED CORRECTLY** in `docs/api.md`

#### Query Method
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:603-676
async def query(
    self,
    query: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] | None = None,
    top_k: int | None = None,
    rerank_top_k: int | None = None,
    enable_rerank: bool | None = None,
    only_context: bool = False,
    system_prompt: str | None = None,
) -> str:
```

✅ **DOCUMENTED CORRECTLY** - Returns `str`, not `dict`

#### Query With Memory Method
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:762-889
async def query_with_memory(
    self,
    query: str,
    session_id: str,
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] | None = None,
    top_k: int | None = None,
    max_history_messages: int = 10,
) -> dict[str, Any]:
```

✅ **DOCUMENTED CORRECTLY** - Parameters match actual signature

#### Create Conversation Session
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:891-907
async def create_conversation_session(
    self,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
```

✅ **DOCUMENTED CORRECTLY**

#### Get Conversation History
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:909-925
async def get_conversation_history(
    self,
    session_id: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
```

✅ **DOCUMENTED CORRECTLY**

#### Clear Conversation
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:927-935
async def clear_conversation(self, session_id: str) -> None:
```

✅ **DOCUMENTED CORRECTLY** - Fixed from incorrect `delete_conversation_session()`

#### Get Status
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:947-980
async def get_status(self) -> dict[str, Any]:
```

✅ **DOCUMENTED CORRECTLY** - Fixed from incorrect `get_stats()`

#### Get Knowledge Base Stats
```python
# ACTUAL CODE: src/hybridrag/core/rag.py:982-1036
async def get_knowledge_base_stats(self) -> dict[str, Any]:
```

✅ **DOCUMENTED CORRECTLY**

### Settings Validation

#### Settings Class Structure
```python
# ACTUAL CODE: src/hybridrag/config/settings.py:14-199
class Settings(BaseSettings):
    mongodb_uri: SecretStr = Field(...)
    mongodb_database: str = Field(default="hybridrag")
    mongodb_workspace: str = Field(default="default")
    
    voyage_api_key: SecretStr | None = Field(default=None)
    voyage_embedding_model: str = Field(default="voyage-3-large")
    voyage_context_model: str = Field(default="voyage-context-3")
    voyage_rerank_model: str = Field(default="rerank-2.5")
    
    llm_provider: Literal["anthropic", "openai", "gemini"] = Field(default="gemini")
    anthropic_api_key: SecretStr | None = Field(default=None)
    anthropic_model: str = Field(default="claude-sonnet-4-20250514")
    openai_api_key: SecretStr | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o")
    gemini_api_key: SecretStr | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")
    
    embedding_provider: Literal["voyage"] = Field(default="voyage")
    
    default_query_mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(default="mix")
    default_top_k: int = Field(default=60, ge=1, le=200)
    default_rerank_top_k: int = Field(default=10, ge=1, le=50)
    enable_rerank: bool = Field(default=True)
    
    enable_implicit_expansion: bool = Field(default=True)
    implicit_expansion_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    implicit_expansion_max: int = Field(default=10, ge=1)
    enable_entity_boosting: bool = Field(default=True)
    entity_boost_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    
    embedding_dim: int = Field(default=1024)
    max_token_size: int = Field(default=4096)
    embedding_batch_size: int = Field(default=128, ge=1, le=128)
    
    max_token_for_text_unit: int = Field(default=4000)
    max_token_for_local_context: int = Field(default=4000)
    max_token_for_global_context: int = Field(default=4000)
    
    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: SecretStr | None = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")
```

✅ **DOCUMENTED CORRECTLY** in `docs/configuration.md` - All fields match actual implementation

### Query Modes Validation

```python
# ACTUAL CODE: src/hybridrag/core/rag.py:28-45
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
```

✅ **DOCUMENTED CORRECTLY** - All 6 modes match:
- `local` ✅
- `global` ✅
- `hybrid` ✅
- `naive` ✅
- `mix` ✅
- `bypass` ✅

### UI Command Validation

```python
# ACTUAL CODE: src/hybridrag/ui/chat.py:1-26
# Chainlit UI entry point
```

```bash
# ACTUAL COMMAND: README.md:153
chainlit run src/hybridrag/ui/chat.py
```

✅ **DOCUMENTED CORRECTLY** in `docs/installation.md` and `README.md`

### Environment Variables Validation

```python
# ACTUAL CODE: src/hybridrag/config/settings.py:17-21
model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)
```

✅ **DOCUMENTED CORRECTLY** - Settings load from `.env` file

Required variables match actual Settings fields:
- `MONGODB_URI` ✅
- `MONGODB_DATABASE` ✅
- `VOYAGE_API_KEY` ✅
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` ✅

## Summary

### Fixed Issues

1. ✅ Changed `ingest()` → `insert()` and `ingest_files()`/`ingest_file()`
2. ✅ Fixed `HybridRAG.create()` → `HybridRAG()` + `initialize()` or `create_hybridrag()`
3. ✅ Fixed `query()` return type: `dict` → `str`
4. ✅ Fixed `query()` parameters: `vector_search_top_k`/`keyword_search_top_k` → `top_k`/`rerank_top_k`
5. ✅ Fixed `delete_conversation_session()` → `clear_conversation()`
6. ✅ Fixed `get_stats()` → `get_status()` and `get_knowledge_base_stats()`
7. ✅ Fixed Settings field names to match actual implementation
8. ✅ Fixed LLM model names (e.g., `gemini-2.5-flash` not `gemini-pro`)

### Verified Correct

- ✅ All query modes match code
- ✅ All Settings fields match code
- ✅ All method signatures match code
- ✅ UI command matches code
- ✅ Environment variable names match code
- ✅ Factory function signature matches code

## Conclusion

All documentation has been validated against the actual codebase. All methods, parameters, return types, and configuration options now match the implementation exactly.

