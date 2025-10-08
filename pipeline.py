"""
Main pipeline script for RAG knowledge graph generation.

This is the entry point for the refactored, production-grade RAG pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import argparse

import config
from orchestrator import ProcessingOrchestrator


def setup_logging():
    """Configure logging for the pipeline."""
    # Create logs directory if it doesn't exist
    config.LOG_DIR.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.LOG_DIR / f"pipeline_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Knowledge Graph Generation Pipeline"
    )

    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Name of the collection to process (e.g., 'spurgeon', 'puritans')"
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Source directory containing files to process"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=config.MAX_WORKERS,
        help=f"Number of parallel workers (default: {config.MAX_WORKERS})"
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=config.CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N files (default: {config.CHECKPOINT_INTERVAL})"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )

    return parser.parse_args()


def main():
    """Main pipeline execution."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging()

    logger.info("=" * 80)
    logger.info("RAG KNOWLEDGE GRAPH GENERATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Source directory: {args.source_dir}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Checkpoint interval: {args.checkpoint_interval}")
    logger.info(f"Resume from checkpoint: {args.resume}")
    logger.info("=" * 80)

    # Update config with command-line arguments
    config.MAX_WORKERS = args.workers
    config.CHECKPOINT_INTERVAL = args.checkpoint_interval

    # Validate source directory
    if not args.source_dir.exists():
        logger.error(f"Source directory does not exist: {args.source_dir}")
        sys.exit(1)

    try:
        # Initialize orchestrator
        orchestrator = ProcessingOrchestrator(args.collection)

        # Run pipeline
        orchestrator.run(args.source_dir)

        logger.info("Pipeline completed successfully!")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        logger.info("Progress has been saved to checkpoint")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
