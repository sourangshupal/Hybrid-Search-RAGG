# HybridRAG Interactive Notebooks

Learn HybridRAG through interactive Jupyter notebooks.

## Getting Started

```bash
# Install Jupyter Lab (if not already installed)
pip install jupyterlab ipykernel

# Start Jupyter Lab
make notebooks
# or
jupyter lab notebooks/
```

## Notebooks

### 01. Getting Started
**File:** `01_getting_started.ipynb`
**Duration:** 15 minutes
**Prerequisites:** None

Learn the basics:
- Environment setup
- Connect to MongoDB Atlas
- Ingest sample documents
- Run your first queries
- Visualize results

### 02. Hybrid Search Deep Dive
**File:** `02_hybrid_search_deep_dive.ipynb`
**Duration:** 25 minutes
**Prerequisites:** Notebook 01

Explore search modes:
- Vector-only search
- Keyword-only search
- Hybrid search with weights
- Understanding `$rankFusion`
- Comparing search quality

### 03. Knowledge Graph Exploration
**File:** `03_knowledge_graph_exploration.ipynb`
**Duration:** 30 minutes
**Prerequisites:** Notebook 01

Work with knowledge graphs:
- Building entity graphs
- Graph traversal queries
- Combining graphs with RAG
- Visualizing relationships

### 04. Prompt Engineering
**File:** `04_prompt_engineering.ipynb`
**Duration:** 20 minutes
**Prerequisites:** Notebook 01

Master system prompts:
- Query type detection
- Reranking strategies
- Custom prompt creation
- Response quality optimization

### 05. Performance Tuning
**File:** `05_performance_tuning.ipynb`
**Duration:** 30 minutes
**Prerequisites:** Notebooks 01-02

Optimize for production:
- Search parameter tuning
- Caching strategies
- Batch processing
- Performance benchmarking
- Cost optimization

## Learning Path

```
Start Here
    ↓
01_getting_started.ipynb
    ↓
┌───┴───┐
│       │
02      04
Hybrid  Prompt
Search  Engin.
│       │
└───┬───┘
    ↓
┌───┴───┐
│       │
03      05
Graph   Perf
Expl.   Tuning
```

## Tips

- **Run cells sequentially**: Each notebook builds on previous cells
- **Edit and experiment**: Modify code to learn by doing
- **Check .env**: Ensure your API keys are configured
- **Restart kernel**: If you get import errors, restart the kernel

## Troubleshooting

### MongoDB Connection Errors
```bash
# Check your connection string in .env
cat .env | grep MONGODB_URI

# Test connection
python -c "from hybridrag.config import get_settings; print(get_settings().MONGODB_URI)"
```

### Missing Dependencies
```bash
# Install all dependencies
make install-all
# or
pip install -e ".[all]"
```

### Jupyter Not Found
```bash
# Install Jupyter Lab
pip install jupyterlab ipykernel

# Register kernel
python -m ipykernel install --user --name=hybridrag
```

## Resources

- [HybridRAG Documentation](../README.md)
- [API Reference](../docs/)
- [Examples](../examples/)
- [MongoDB Atlas Setup](https://www.mongodb.com/docs/atlas/)

## Contributing

Found an issue or have a suggestion? Open an issue or PR!
