# RAG Knowledge Graph Pipeline - Production Version

A production-grade, refactored pipeline for building knowledge graphs from theological texts.

## Architecture

The pipeline has been refactored into a modular, maintainable architecture:

```
spurgeon-rag-app/
├── config.py              # Centralized configuration
├── text_processor.py      # Text extraction and chunking
├── graph_builder.py       # Graph construction and persistence
├── orchestrator.py        # Parallel processing orchestration
├── pipeline.py            # Main entry point
└── logs/                  # Processing logs
```

## Key Improvements

### 1. **Performance Optimizations**
- **Shared spaCy Model**: The spaCy model is loaded once and shared across all worker processes, eliminating the 7x redundant loading issue
- **Streaming/Generators**: Text processing uses generators to minimize memory usage
- **Efficient Checkpointing**: Atomic writes prevent corruption

### 2. **Memory Management**
- Generator-based chunk processing
- Incremental graph merging
- Controlled worker pool size

### 3. **Error Handling**
- Robust try/except blocks in worker processes
- Individual file failures don't stop the pipeline
- Detailed error logging with stack traces
- Automatic checkpoint recovery

### 4. **Code Organization**
- **TextProcessor**: Handles all text extraction and chunking
- **GraphBuilder**: Manages graph construction and entity extraction
- **ProcessingOrchestrator**: Coordinates parallel processing and checkpointing
- **Config module**: All parameters in one place

## Configuration

Edit [config.py](config.py) to adjust parameters:

```python
# Processing
MAX_WORKERS = 7
CHECKPOINT_INTERVAL = 50

# Chunking
PARENT_CHUNK_SIZE = 1500
CHILD_CHUNK_SIZE = 512

# Entity extraction
MIN_ENTITY_FREQUENCY = 2
SIMILARITY_THRESHOLD = 0.85
```

## Usage

### Basic Usage

```bash
python pipeline.py \
  --collection spurgeon \
  --source-dir "C:/Users/danieo/Downloads/TheologyLibrary/spurgeon"
```

### Advanced Options

```bash
python pipeline.py \
  --collection puritans \
  --source-dir "C:/Users/danieo/Downloads/TheologyLibrary/puritans" \
  --workers 10 \
  --checkpoint-interval 25 \
  --resume
```

### Command-Line Arguments

- `--collection`: Name of the collection (required)
- `--source-dir`: Directory containing source files (required)
- `--workers`: Number of parallel workers (default: 7)
- `--checkpoint-interval`: Files between checkpoints (default: 50)
- `--resume`: Resume from checkpoint if available

## Checkpointing & Recovery

The pipeline automatically:
- Saves checkpoints every N files (configurable)
- Tracks processed files to avoid reprocessing
- Resumes from last checkpoint on restart
- Uses atomic writes to prevent corruption

### Checkpoint Files

- `checkpoint_<collection>.gml` - Graph checkpoint
- `processed_files_<collection>.txt` - List of processed files
- `graph_<collection>.gml` - Final output graph

## Logging

Logs are saved to `logs/pipeline_<timestamp>.log` and include:
- Processing progress
- Graph statistics
- Error details with stack traces
- Performance metrics

## Output

The pipeline produces:
- **Final Graph**: `graph_<collection>.gml`
- **Statistics**:
  - Total nodes (chunks + entities)
  - Total edges (relationships)
  - Entity nodes count
  - Processing time

## Example Workflow

### 1. Process Spurgeon Collection

```bash
python pipeline.py \
  --collection spurgeon \
  --source-dir "C:/Users/danieo/Downloads/TheologyLibrary/spurgeon"
```

### 2. Process Additional Collections

```bash
python pipeline.py \
  --collection Epub_Vol_1-25 \
  --source-dir "C:/Users/danieo/Downloads/TheologyLibrary/Epub Vol. 1-25"
```

### 3. Merge All Graphs

```bash
python merge_graphs.py
```

This creates `puritan_master_graph.gml` combining all collections.

## Error Handling

The pipeline handles errors gracefully:

1. **File-level errors**: Skip bad files, continue processing
2. **Worker crashes**: Logged with full stack trace
3. **Checkpoint corruption**: Prevented by atomic writes
4. **User interruption**: Progress saved automatically

## Performance Tips

1. **Adjust workers**: Set `--workers` based on your CPU cores
2. **Checkpoint frequency**: More frequent = safer but slower
3. **Memory constraints**: Reduce `MAX_WORKERS` if running out of RAM
4. **SSD recommended**: For faster file I/O during checkpointing

## Monitoring Progress

Watch the log file in real-time:

```bash
tail -f logs/pipeline_<timestamp>.log
```

Or use the console output which shows:
- Files processed count
- Current graph statistics
- Checkpoint saves
- Error messages

## Troubleshooting

### spaCy Model Missing

```bash
python -m spacy download en_core_web_sm
```

### Memory Issues

Reduce workers in config:
```python
MAX_WORKERS = 4  # Instead of 7
```

### Slow Processing

- Use SSD for checkpoint files
- Increase checkpoint interval
- Reduce entity extraction complexity

## Technical Details

### Graph Structure

**Nodes**:
- Chunk nodes (parent/child with text content)
- Entity nodes (theological concepts, biblical books, figures)

**Edges**:
- `has_child`: Parent → Child chunk
- `mentions`: Chunk → Entity
- `co_occurs_with`: Entity ↔ Entity (weighted)

### Entity Extraction

The pipeline extracts:
- **spaCy NER**: PERSON, ORG, GPE, EVENT, WORK_OF_ART
- **Theological concepts**: 40+ predefined terms
- **Biblical books**: All 66 books
- **Historical figures**: Key theologians and biblical figures

### Parallel Processing

- Uses `ProcessPoolExecutor` for true parallelism
- spaCy model shared via worker initializer
- Results merged incrementally to main graph
- Checkpoint saves don't block workers

## Migration from Old Script

If you have existing checkpoints from the old monolithic script:

1. They are compatible with the new pipeline
2. Run with `--resume` to continue processing
3. The new pipeline will complete remaining files
4. Output format is identical

## Future Enhancements

Possible improvements:
- [ ] Distributed processing across machines
- [ ] Database backend instead of GML files
- [ ] Real-time progress web dashboard
- [ ] Entity disambiguation and linking
- [ ] Semantic similarity clustering
- [ ] Interactive graph visualization

## Support

For issues or questions:
1. Check the log files first
2. Review configuration settings
3. Ensure all dependencies are installed
4. Verify source directory paths

---

**Version**: 2.0 (Production Refactor)
**Date**: 2025-10-08
