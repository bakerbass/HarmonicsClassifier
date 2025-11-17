# Agent Task: Build a Parser for the IDMT-SMT-GUITAR Dataset  
A clean, properly formatted Markdown file including fenced code blocks.

---

## 1. Goal
Write Python code to parse the **IDMT-SMT-GUITAR** dataset and produce a machine‑learning‑ready dataset for:
- Long sustained natural harmonics  
- Dead notes  
- General notes  

Use XML annotations + audio to generate:
- A metadata table (CSV/Parquet)  
- Optional per‑note audio snippets  

---

## 2. Dataset Summary
Datasets 1–3 include XML note‑level annotations with fields such as:

- `onsetSec`, `offsetSec`
- `pitch`
- `stringNumber`, `fretNumber`
- `excitationStyle`
- `expressionStyle`

Expression styles relevant to your classifier:

- `HA` — natural harmonics  
- `DN` — dead notes  
- `NO` and others — general notes  

---

## 3. Requirements
Primary Python stack:

```python
import os
from pathlib import Path
import pandas as pd
import soundfile as sf
import librosa
import xml.etree.ElementTree as ET
```

Input:
- Dataset root with WAV + XML files

Output:
- Metadata table  
- Optional audio snippets  

---

## 4. File Discovery Logic
Walk the root directory, collecting WAV/XML pairs by shared basenames:

```python
def find_pairs(root):
    audio_files = {}
    xml_files = {}

    for path in Path(root).rglob("*"):
        if path.suffix.lower() == ".wav":
            audio_files[path.stem] = path
        elif path.suffix.lower() == ".xml":
            xml_files[path.stem] = path

    pairs = []
    for stem, wavpath in audio_files.items():
        if stem in xml_files:
            pairs.append((wavpath, xml_files[stem]))

    return pairs
```

Subset identification:
- Derive `subset_id` from directory names when possible.  
- Allow a fallback if names do not match expected patterns.

---

## 5. XML Parsing
Each `<event>` provides note information. Extract fields robustly:

```python
def parse_xml_events(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    events = []
    for ev in root.findall(".//event"):
        def get(tag):
            node = ev.find(tag)
            return node.text.strip() if node is not None and node.text else None

        onset = float(get("onsetSec"))
        offset = float(get("offsetSec"))
        pitch = int(get("pitch"))

        events.append({
            "onset_sec": onset,
            "offset_sec": offset,
            "duration_sec": offset - onset,
            "pitch_midi": pitch,
            "excitation_style": get("excitationStyle"),
            "expression_style": get("expressionStyle"),
            "string_number": get("stringNumber"),
            "fret_number": get("fretNumber"),
        })

    return events
```

---

## 6. Label Definition
Derived labels:

```python
def derive_labels(ev):
    expr = ev["expression_style"]

    is_harm = expr == "HA"
    is_dead = expr == "DN"
    is_gen  = not is_harm and not is_dead

    if is_harm:
        cat = "harmonic"
    elif is_dead:
        cat = "dead_note"
    else:
        cat = "general_note"

    ev.update({
        "is_harmonic": is_harm,
        "is_dead_note": is_dead,
        "is_general_note": is_gen,
        "label_category": cat,
    })
    return ev
```

---

## 7. Audio Snippet Extraction

```python
def extract_snippet(audio_path, onset, offset, pad=0.02):
    audio, sr = librosa.load(audio_path, sr=None)

    start = max(0, int((onset - pad) * sr))
    end   = min(len(audio), int((offset + pad) * sr))

    return audio[start:end], sr
```

Save snippets deterministically:

```python
def save_snippet(audio, sr, outpath):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(outpath, audio, sr)
```

Naming convention:

```
subset{subset_id}_{basename}_event{index:04d}_pitch{pitch}_expr{expr}.wav
```

---

## 8. Project Structure

Recommended module layout:

```
dataset_builder/
    config.py
    file_discovery.py
    xml_parser.py
    labeling.py
    audio_snippets.py
    build_dataset.py
run_build_dataset.py
```

Main script (`run_build_dataset.py`):

```python
if __name__ == "__main__":
    import argparse
    from build_dataset import build

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--export-snippets", action="store_true")
    args = parser.parse_args()

    build(args.data_root, args.output_dir, args.export_snippets)
```

---

## 9. Sanity Checks
Record statistics:

```python
print(df["label_category"].value_counts())
```

Checks:
- WAV/XML mismatch counts  
- Bad durations (< 10ms)  
- Distribution of expression styles  

---

## 10. Extensibility
Future options:
- Extract only long‑sustain harmonics  
- Detect special techniques (slides, bends)  
- Add guitar‑string/fret filtering  
- Extend Dataset 4 parsing  

---

This file is intended to be given directly to an autonomous coding agent as a complete project specification for parsing and organizing the IDMT-SMT-GUITAR dataset.
