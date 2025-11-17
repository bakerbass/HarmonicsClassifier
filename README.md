# IDMT-SMT-GUITAR Dataset Parser

A modular Python parser for extracting and organizing annotated guitar note data from the IDMT-SMT-GUITAR dataset. This tool creates a machine-learning-ready dataset with metadata and optional audio snippets.

## Features

- **Automated Discovery**: Finds matching WAV/XML file pairs in the dataset
- **XML Parsing**: Extracts note-level annotations (onset, offset, pitch, style, etc.)
- **Label Derivation**: Classifies notes as harmonics, dead notes, or general notes
- **Audio Extraction**: Optional extraction of individual note audio snippets
- **Metadata Export**: Generates comprehensive CSV/Parquet metadata files
- **Statistics & Validation**: Reports dataset statistics and runs sanity checks

## Installation

1. Clone or download this project
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Structure

The parser expects the IDMT-SMT-GUITAR dataset with the following structure:

```
IDMT-SMT-GUITAR/
├── dataset1/
│   ├── audio_file1.wav
│   ├── audio_file1.xml
│   ├── audio_file2.wav
│   ├── audio_file2.xml
│   └── ...
├── dataset2/
└── dataset3/
```

## Usage

### Basic Usage (Metadata Only)

```bash
python run_build_dataset.py --data-root ./IDMT-SMT-GUITAR --output-dir ./processed_dataset
```

### With Audio Snippet Extraction

```bash
python run_build_dataset.py --data-root ./IDMT-SMT-GUITAR --output-dir ./processed_dataset --export-snippets
```

### With Sanity Checks

```bash
python run_build_dataset.py --data-root ./IDMT-SMT-GUITAR --output-dir ./processed_dataset --sanity-check
```

### Quiet Mode

```bash
python run_build_dataset.py --data-root ./IDMT-SMT-GUITAR --output-dir ./processed_dataset --quiet
```

## Command-Line Arguments

- `--data-root`: Root directory containing the IDMT-SMT-GUITAR dataset (required)
- `--output-dir`: Output directory for processed data and metadata (required)
- `--export-snippets`: Extract individual audio snippets for each note
- `--sanity-check`: Run validation checks on the parsed dataset
- `--quiet`: Suppress progress output

## Output Structure

### Without Snippets

```
processed_dataset/
├── metadata.csv
└── metadata.parquet
```

### With Snippets

```
processed_dataset/
├── metadata.csv
├── metadata.parquet
└── snippets/
    ├── harmonic/
    │   ├── subset1_file1_event0001_pitch64_exprHA.wav
    │   └── ...
    ├── dead_note/
    │   ├── subset1_file2_event0003_pitch60_exprDN.wav
    │   └── ...
    └── general_note/
        ├── subset1_file3_event0002_pitch67_exprNO.wav
        └── ...
```

## Metadata Fields

The generated CSV/Parquet files contain the following fields:

- `subset_id`: Dataset subset identifier
- `source_audio`: Path to source WAV file
- `source_annotation`: Path to source XML file
- `event_index`: Event number within the file
- `onset_sec`: Note start time (seconds)
- `offset_sec`: Note end time (seconds)
- `duration_sec`: Note duration (seconds)
- `pitch_midi`: MIDI pitch number
- `string_number`: Guitar string number (if available)
- `fret_number`: Fret number (if available)
- `excitation_style`: Excitation technique
- `expression_style`: Expression style code (HA, DN, NO, etc.)
- `label_category`: Derived label (harmonic, dead_note, general_note)
- `is_harmonic`: Boolean flag for harmonics
- `is_dead_note`: Boolean flag for dead notes
- `is_general_note`: Boolean flag for general notes
- `snippet_path`: Path to extracted audio snippet (if exported)

## Label Categories

- **harmonic**: Natural harmonics (expression style = HA)
- **dead_note**: Dead/muted notes (expression style = DN)
- **general_note**: All other notes (NO and other styles)

## Module Structure

```
dataset_builder/
├── __init__.py
├── config.py              # Configuration settings
├── file_discovery.py      # WAV/XML pair discovery
├── xml_parser.py          # XML annotation parsing
├── labeling.py            # Label derivation logic
├── audio_snippets.py      # Audio extraction utilities
└── build_dataset.py       # Main orchestration module
```

## Example Python Usage

```python
from dataset_builder.build_dataset import build
import pandas as pd

# Build dataset
df = build(
    data_root="./IDMT-SMT-GUITAR",
    output_dir="./processed_dataset",
    export_snippets=True,
    verbose=True
)

# Filter harmonics only
harmonics = df[df["is_harmonic"] == True]
print(f"Found {len(harmonics)} harmonic notes")

# Get label distribution
print(df["label_category"].value_counts())
```

## Customization

Edit `dataset_builder/config.py` to customize:

- Snippet padding duration
- Minimum note duration threshold
- Output file naming conventions
- Label mappings

## Troubleshooting

**No file pairs found**: Ensure your dataset directory structure matches the expected format with matching WAV/XML basenames.

**Import errors**: Make sure all dependencies are installed via `pip install -r requirements.txt`.

**Memory issues with large datasets**: Process without `--export-snippets` first to create metadata only, then extract snippets in batches if needed.

## License

This parser is provided as-is for working with the IDMT-SMT-GUITAR dataset. Please refer to the original dataset's license for usage terms.

## Citation

If you use the IDMT-SMT-GUITAR dataset, please cite the original paper:

```
Kehling, C., Abeßer, J., Dittmar, C., & Schuller, G. (2014).
Automatic tablature transcription of electric guitar recordings by estimation of score-and instrument-related parameters.
```
