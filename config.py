"""
Configuration file for the RAG pipeline.
All parameters and settings are centralized here for easy maintenance.
"""

from pathlib import Path

# ==============================================================================
# DIRECTORY PATHS
# ==============================================================================
LIBRARY_BASE_DIR = Path(r"C:\Users\danieo\Downloads\TheologyLibrary")
PURITANS_DIR = LIBRARY_BASE_DIR / "puritans"
SPURGEON_DIR = LIBRARY_BASE_DIR / "spurgeon"

# ==============================================================================
# PROCESSING PARAMETERS
# ==============================================================================
# Parallel processing
MAX_WORKERS = 7  # Number of parallel workers

# Checkpointing
CHECKPOINT_INTERVAL = 50  # Save checkpoint every N files

# Text chunking
PARENT_CHUNK_SIZE = 1500  # Characters per parent chunk
PARENT_CHUNK_OVERLAP = 200  # Overlap between parent chunks
CHILD_CHUNK_SIZE = 512  # Characters per child chunk
CHILD_CHUNK_OVERLAP = 50  # Overlap between child chunks

# Entity extraction
MIN_ENTITY_FREQUENCY = 2  # Minimum frequency for entity to be included
SIMILARITY_THRESHOLD = 0.85  # Threshold for entity similarity matching

# ==============================================================================
# SPACY MODEL
# ==============================================================================
SPACY_MODEL = "en_core_web_sm"

# ==============================================================================
# ENTITY CATEGORIES
# ==============================================================================
# Theological concepts to extract
THEOLOGICAL_CONCEPTS = {
    "salvation", "grace", "faith", "justification", "sanctification",
    "redemption", "atonement", "regeneration", "adoption", "glorification",
    "election", "predestination", "covenant", "trinity", "incarnation",
    "resurrection", "propitiation", "reconciliation", "imputation",
    "righteousness", "holiness", "sin", "repentance", "conversion",
    "perseverance", "assurance", "prayer", "worship", "sacraments",
    "baptism", "communion", "church", "ministry", "preaching",
    "scripture", "revelation", "inspiration", "authority", "gospel",
    "law", "gospel", "kingdom", "eschatology", "judgment"
}

# Biblical books to extract
BIBLICAL_BOOKS = {
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles",
    "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel",
    "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk",
    "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts", "Romans",
    "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John",
    "Jude", "Revelation"
}

# Historical figures to extract
HISTORICAL_FIGURES = {
    "Augustine", "Calvin", "Luther", "Spurgeon", "Owen",
    "Edwards", "Bunyan", "Baxter", "Whitefield", "Wesley",
    "Wycliffe", "Tyndale", "Knox", "Zwingli", "Melanchthon",
    "Athanasius", "Chrysostom", "Aquinas", "Anselm",
    "Paul", "Peter", "John", "Moses", "David", "Abraham",
    "Isaiah", "Jeremiah", "Ezekiel", "Daniel"
}

# ==============================================================================
# LOGGING
# ==============================================================================
LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ==============================================================================
# FILE PROCESSING
# ==============================================================================
SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".txt"}
ENCODING = "utf-8"

# ==============================================================================
# GRAPH SETTINGS
# ==============================================================================
GRAPH_FORMAT = "gml"  # Output format for graphs
CHECKPOINT_PREFIX = "checkpoint_"
FINAL_GRAPH_PREFIX = "graph_"
