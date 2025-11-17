"""
Extract note clips from full audio files.

This script reads the metadata CSV, extracts clips for specified note types
(harmonics, dead notes, general notes), and saves them in a structured
directory organized by note type.
"""

import argparse
import pandas as pd
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def load_metadata(metadata_path: str, note_types: list = None, balance_classes: bool = False, max_per_class: int = None) -> pd.DataFrame:
    """
    Load and filter metadata for specified note types.
    
    Args:
        metadata_path: Path to metadata CSV
        note_types: List of note types to extract. If None, extract all types.
                   Options: ['harmonic', 'dead_note', 'general_note']
        balance_classes: If True, balance classes to match the smallest class
        max_per_class: Maximum samples per class. If None and balance_classes is True,
                      uses the count of the smallest class
    """
    df = pd.read_csv(metadata_path)
    
    if note_types is None or 'all' in note_types:
        filtered_df = df.copy()
        print(f"Found {len(filtered_df)} total notes in dataset")
    else:
        filtered_df = df[df['label_category'].isin(note_types)].copy()
        print(f"Found {len(filtered_df)} notes of types: {note_types}")
    
    # Print initial breakdown by type
    type_counts = filtered_df['label_category'].value_counts()
    print("\nInitial distribution:")
    for note_type, count in type_counts.items():
        print(f"  - {note_type}: {count}")
    
    # Balance classes if requested
    if balance_classes or max_per_class is not None:
        if max_per_class is None:
            # Use the smallest class size
            max_per_class = type_counts.min()
        
        print(f"\nBalancing classes to max {max_per_class} samples per class...")
        
        balanced_dfs = []
        for note_type in filtered_df['label_category'].unique():
            type_df = filtered_df[filtered_df['label_category'] == note_type]
            
            if len(type_df) > max_per_class:
                # Randomly sample to balance
                type_df = type_df.sample(n=max_per_class, random_state=42)
            
            balanced_dfs.append(type_df)
        
        filtered_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # Print balanced distribution
        balanced_counts = filtered_df['label_category'].value_counts()
        print("\nBalanced distribution:")
        for note_type, count in balanced_counts.items():
            print(f"  - {note_type}: {count}")
    
    return filtered_df


def group_by_audio_file(df: pd.DataFrame) -> dict:
    """Group note events by source audio file and label category."""
    grouped = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        grouped[row['source_audio']][row['label_category']].append(row)
    return grouped


def extract_note_clips(
    audio_path: str,
    events_by_type: dict,
    output_dir: Path,
    padding_sec: float = 0.5,
    min_gap_sec: float = 0.2
):
    """
    Extract clips containing notes from an audio file, organized by type.
    
    Args:
        audio_path: Path to source audio file
        events_by_type: Dictionary mapping note types to lists of event rows
        output_dir: Output directory for clips
        padding_sec: Padding before/after each note
        min_gap_sec: Minimum gap between notes to keep them separate
    """
    # Load audio
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return
    
    audio_stem = Path(audio_path).stem
    
    # Process each note type separately
    for note_type, events in events_by_type.items():
        if len(events) == 0:
            continue
        
        # Create subdirectory for this note type
        type_output_dir = output_dir / note_type
        type_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sort events by onset time
        events = sorted(events, key=lambda x: x['onset_sec'])
        
        # Merge overlapping or close events into clips
        clips = []
        current_clip_start = None
        current_clip_end = None
        current_clip_events = []
        
        for event in events:
            onset = event['onset_sec']
            offset = event['offset_sec']
            
            # Add padding
            clip_start = max(0, onset - padding_sec)
            clip_end = min(len(audio) / sr, offset + padding_sec)
            
            if current_clip_start is None:
                # First clip
                current_clip_start = clip_start
                current_clip_end = clip_end
                current_clip_events = [event]
            elif clip_start - current_clip_end < min_gap_sec:
                # Merge with current clip
                current_clip_end = clip_end
                current_clip_events.append(event)
            else:
                # Save current clip and start new one
                clips.append((current_clip_start, current_clip_end, current_clip_events))
                current_clip_start = clip_start
                current_clip_end = clip_end
                current_clip_events = [event]
        
        # Don't forget the last clip
        if current_clip_start is not None:
            clips.append((current_clip_start, current_clip_end, current_clip_events))
        
        # Extract and save clips
        for clip_idx, (start_sec, end_sec, clip_events) in enumerate(clips):
            # Extract audio segment
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            clip_audio = audio[start_sample:end_sample]
            
            # Generate filename
            num_notes = len(clip_events)
            pitches = [str(e['pitch_midi']) for e in clip_events]
            pitch_str = "_".join(pitches)
            
            filename = f"{audio_stem}_clip{clip_idx:02d}_{note_type}_n{num_notes}_pitches{pitch_str}.wav"
            output_path = type_output_dir / filename
            
            # Save clip
            sf.write(output_path, clip_audio, sr)
            
            # Print info
            print(f"  [{note_type}] Clip {clip_idx + 1}: {start_sec:.2f}s - {end_sec:.2f}s ({num_notes} notes) -> {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract audio clips for guitar notes organized by type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all note types
  python extract_harmonic_clips.py --note-types all

  # Extract only harmonics and dead notes
  python extract_harmonic_clips.py --note-types harmonic dead_note

  # Custom output directory and padding
  python extract_harmonic_clips.py --output-dir ./note_clips --padding 1.0

  # Merge close notes (within 0.5s)
  python extract_harmonic_clips.py --min-gap 0.5
        """
    )
    
    parser.add_argument(
        "--metadata",
        default="./processed_dataset/metadata.csv",
        help="Path to metadata CSV file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./note_clips",
        help="Output directory for note clips (subdirectories will be created for each type)"
    )
    
    parser.add_argument(
        "--note-types",
        nargs='+',
        default=['all'],
        choices=['all', 'harmonic', 'dead_note', 'general_note'],
        help="Types of notes to extract"
    )
    
    parser.add_argument(
        "--padding",
        type=float,
        default=0.001,
        help="Padding (in seconds) before/after each note"
    )
    
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.01,
        help="Minimum gap (in seconds) between notes to keep them in separate clips"
    )
    
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance classes to have equal representation (uses smallest class count)"
    )
    
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum number of samples per class (enables balancing)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load metadata
        print("=" * 60)
        print("Guitar Note Clip Extractor")
        print("=" * 60)
        print(f"Metadata: {args.metadata}")
        print(f"Output: {args.output_dir}")
        print(f"Note types: {args.note_types}")
        print(f"Padding: {args.padding}s")
        print(f"Min gap: {args.min_gap}s")
        print(f"Balance classes: {args.balance or args.max_per_class is not None}")
        if args.max_per_class:
            print(f"Max per class: {args.max_per_class}")
        print()
        
        note_types = None if 'all' in args.note_types else args.note_types
        notes_df = load_metadata(
            args.metadata, 
            note_types, 
            balance_classes=args.balance,
            max_per_class=args.max_per_class
        )
        
        if len(notes_df) == 0:
            print("No notes found matching the specified types!")
            return
        
        # Group by source audio file
        grouped = group_by_audio_file(notes_df)
        print(f"\nNotes found in {len(grouped)} audio files")
        print()
        
        # Process each audio file
        output_path = Path(args.output_dir)
        
        for audio_file, events_by_type in tqdm(grouped.items(), desc="Processing audio files"):
            total_events = sum(len(events) for events in events_by_type.values())
            print(f"\n{audio_file} ({total_events} notes)")
            extract_note_clips(
                audio_file,
                events_by_type,
                output_path,
                padding_sec=args.padding,
                min_gap_sec=args.min_gap
            )
        
        print()
        print("=" * 60)
        print(f"✓ Extraction complete! Clips saved to: {args.output_dir}")
        print("\nDirectory structure:")
        for note_type in notes_df['label_category'].unique():
            type_dir = output_path / note_type
            if type_dir.exists():
                num_files = len(list(type_dir.glob('*.wav')))
                print(f"  {note_type}/  ({num_files} clips)")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
