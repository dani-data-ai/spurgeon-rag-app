"""
Graph building module for RAG pipeline.
Handles node/edge creation and graph persistence.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
import networkx as nx
import spacy
from spacy.matcher import PhraseMatcher

import config

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Handles knowledge graph construction and persistence."""

    def __init__(self, nlp):
        """
        Initialize the graph builder.

        Args:
            nlp: Shared spaCy model instance
        """
        self.nlp = nlp
        self.graph = nx.Graph()
        self.entity_counter = Counter()

        # Initialize phrase matchers for custom entities
        self._init_phrase_matchers()

    def _init_phrase_matchers(self):
        """Initialize phrase matchers for theological concepts, books, and figures."""
        self.theological_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.book_matcher = PhraseMatcher(self.nlp.vocab)
        self.figure_matcher = PhraseMatcher(self.nlp.vocab)

        # Add patterns
        theological_patterns = [self.nlp.make_doc(term) for term in config.THEOLOGICAL_CONCEPTS]
        self.theological_matcher.add("THEOLOGICAL", theological_patterns)

        book_patterns = [self.nlp.make_doc(book) for book in config.BIBLICAL_BOOKS]
        self.book_matcher.add("BOOK", book_patterns)

        figure_patterns = [self.nlp.make_doc(name) for name in config.HISTORICAL_FIGURES]
        self.figure_matcher.add("FIGURE", figure_patterns)

    def extract_entities_from_text(self, text: str) -> Set[str]:
        """
        Extract entities from text using spaCy NER and custom matchers.

        Args:
            text: Text to extract entities from

        Returns:
            Set of extracted entity strings
        """
        entities = set()

        try:
            # Process text with spaCy
            doc = self.nlp(text[:1000000])  # Limit text length to avoid memory issues

            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "EVENT", "WORK_OF_ART"}:
                    entities.add(ent.text)

            # Extract custom entities
            for match_id, start, end in self.theological_matcher(doc):
                entities.add(doc[start:end].text)

            for match_id, start, end in self.book_matcher(doc):
                entities.add(doc[start:end].text)

            for match_id, start, end in self.figure_matcher(doc):
                entities.add(doc[start:end].text)

        except Exception as e:
            logger.error(f"Error extracting entities: {e}", exc_info=True)

        return entities

    def add_chunk_node(
        self,
        chunk_id: str,
        chunk_text: str,
        chunk_type: str,
        source_file: str,
        parent_id: str = None
    ):
        """
        Add a chunk node to the graph.

        Args:
            chunk_id: Unique identifier for the chunk
            chunk_text: Text content of the chunk
            chunk_type: Type of chunk ('parent' or 'child')
            source_file: Source file name
            parent_id: Parent chunk ID (for child chunks)
        """
        self.graph.add_node(
            chunk_id,
            text=chunk_text,
            type=chunk_type,
            source=source_file
        )

        # If this is a child chunk, link it to its parent
        if parent_id:
            self.graph.add_edge(parent_id, chunk_id, relation="has_child")

    def add_entity_relationships(
        self,
        chunk_id: str,
        entities: Set[str]
    ):
        """
        Add entity nodes and relationships to the graph.

        Args:
            chunk_id: Chunk ID to link entities to
            entities: Set of entity strings
        """
        for entity in entities:
            # Count entity frequency
            self.entity_counter[entity] += 1

            # Only add entity if it meets frequency threshold
            if self.entity_counter[entity] >= config.MIN_ENTITY_FREQUENCY:
                # Add entity node if it doesn't exist
                if entity not in self.graph:
                    self.graph.add_node(entity, type="entity")

                # Add edge from chunk to entity
                self.graph.add_edge(chunk_id, entity, relation="mentions")

    def add_entity_cooccurrence(
        self,
        entities: Set[str]
    ):
        """
        Add co-occurrence relationships between entities.

        Args:
            entities: Set of entities that co-occur
        """
        entity_list = list(entities)

        for i, entity1 in enumerate(entity_list):
            for entity2 in entity_list[i + 1:]:
                # Only add if both entities meet frequency threshold
                if (self.entity_counter[entity1] >= config.MIN_ENTITY_FREQUENCY and
                    self.entity_counter[entity2] >= config.MIN_ENTITY_FREQUENCY):

                    # Check if edge exists
                    if self.graph.has_edge(entity1, entity2):
                        # Increment co-occurrence count
                        self.graph.edges[entity1, entity2]["weight"] = \
                            self.graph.edges[entity1, entity2].get("weight", 1) + 1
                    else:
                        # Add new co-occurrence edge
                        self.graph.add_edge(
                            entity1,
                            entity2,
                            relation="co_occurs_with",
                            weight=1
                        )

    def get_stats(self) -> Dict[str, int]:
        """
        Get graph statistics.

        Returns:
            Dictionary with node and edge counts
        """
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entities": sum(1 for _, data in self.graph.nodes(data=True)
                          if data.get("type") == "entity")
        }

    def save_checkpoint(self, checkpoint_path: Path):
        """
        Save graph checkpoint with atomic write operation.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        try:
            # Save to temporary file first
            temp_path = checkpoint_path.with_suffix('.tmp')
            nx.write_gml(self.graph, str(temp_path))

            # Atomic rename
            shutil.move(str(temp_path), str(checkpoint_path))

            logger.info(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise

    def load_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Load graph from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if successful, False otherwise
        """
        try:
            if checkpoint_path.exists():
                self.graph = nx.read_gml(str(checkpoint_path))
                logger.info(f"Checkpoint loaded: {checkpoint_path}")
                return True
            return False

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return False

    def merge_graph(self, other_graph: nx.Graph):
        """
        Merge another graph into this one.

        Args:
            other_graph: Graph to merge
        """
        self.graph = nx.compose(self.graph, other_graph)

    def save_final_graph(self, output_path: Path):
        """
        Save final graph with atomic write operation.

        Args:
            output_path: Path to save final graph
        """
        try:
            # First save as checkpoint
            checkpoint_path = output_path.with_name(
                f"{config.CHECKPOINT_PREFIX}{output_path.name}"
            )
            self.save_checkpoint(checkpoint_path)

            # Then copy to final location
            shutil.copy(str(checkpoint_path), str(output_path))

            logger.info(f"Final graph saved: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save final graph: {e}", exc_info=True)
            raise
