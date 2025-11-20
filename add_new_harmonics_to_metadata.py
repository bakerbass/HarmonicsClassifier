"""
Add new harmonic recordings to metadata.csv

This script adds entries for the newly recorded harmonic files in note_clips/harmonic/
that match the pattern GB_NH_harmonic_*_pitches*_rep*.wav
"""

import pandas as pd
import re
from pathlib import Path
import librosa

# Paths
METADATA_PATH = Path('processed_dataset/metadata.csv')
HARMONICS_DIR = Path('note_clips/harmonic')

# Load existing metadata
print("Loading existing metadata...")
df = pd.read_csv(METADATA_PATH)
print(f"Current metadata has {len(df)} entries")

# Find new harmonic files (GB_NH prefix pattern)
print("\nFinding new harmonic files...")
new_files = list(HARMONICS_DIR.glob('GB_NH_harmonic_*.wav'))
print(f"Found {len(new_files)} new harmonic files")

if len(new_files) == 0:
    print("No new files to add!")
    exit()

# Parse and create new entries
new_entries = []
# Updated pattern to match both with and without _rep suffix
pattern = re.compile(r'GB_NH_harmonic_n\d+_pitches(\d+)(?:_rep\d+)?\.wav')

for file_path in new_files:
    # Extract pitch from filename
    match = pattern.match(file_path.name)
    if not match:
        print(f"Warning: Could not parse filename: {file_path.name}")
        continue
    
    pitch_midi = int(match.group(1))
    
    # Get audio duration
    try:
        duration = librosa.get_duration(path=str(file_path))
    except Exception as e:
        print(f"Warning: Could not get duration for {file_path.name}: {e}")
        duration = 3.0  # Default duration
    
    # Create relative path
    relative_path = f"note_clips/harmonic/{file_path.name}"
    
    # Create entry
    entry = {
        'subset_id': 'user_recordings_2025_11_18',
        'source_audio': relative_path,
        'source_annotation': '',  # No annotation file
        'event_index': 0,  # Single event per file
        'onset_sec': 0.0,  # Start of file
        'offset_sec': duration,
        'duration_sec': duration,
        'pitch_midi': pitch_midi,
        'string_number': -1,  # Unknown
        'fret_number': -1,  # Unknown
        'excitation_style': 'P',  # Pick
        'expression_style': '',  # No specific expression
        'label_category': 'harmonic',
        'is_harmonic': True,
        'is_dead_note': False,
        'is_general_note': False
    }
    new_entries.append(entry)

print(f"\nParsed {len(new_entries)} entries successfully")

# Create DataFrame from new entries
new_df = pd.DataFrame(new_entries)

# Append to existing metadata
df_combined = pd.concat([df, new_df], ignore_index=True)

# Save updated metadata
print(f"\nSaving updated metadata with {len(df_combined)} total entries...")
df_combined.to_csv(METADATA_PATH, index=False)

print("\n✓ Metadata updated successfully!")
print(f"  Previous entries: {len(df)}")
print(f"  New entries added: {len(new_entries)}")
print(f"  Total entries: {len(df_combined)}")

# Show distribution
print("\nClass distribution after update:")
print(df_combined['label_category'].value_counts())
