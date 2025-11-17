"""
Configuration settings for the IDMT-SMT-GUITAR dataset parser.
"""

# Audio processing settings
DEFAULT_SAMPLE_RATE = None  # Use original sample rate from audio files
SNIPPET_PADDING_SEC = 0.02  # Padding before/after note boundaries

# Label categories
LABEL_HARMONIC = "harmonic"
LABEL_DEAD_NOTE = "dead_note"
LABEL_GENERAL_NOTE = "general_note"

# Expression style mappings
EXPR_HARMONIC = "HA"
EXPR_DEAD_NOTE = "DN"

# Output settings
METADATA_FILENAME = "metadata.csv"
METADATA_PARQUET = "metadata.parquet"

# File naming template for snippets
SNIPPET_NAME_TEMPLATE = "subset{subset_id}_{basename}_event{index:04d}_pitch{pitch}_expr{expr}.wav"

# Validation thresholds
MIN_DURATION_SEC = 0.01  # Minimum note duration (10ms)
