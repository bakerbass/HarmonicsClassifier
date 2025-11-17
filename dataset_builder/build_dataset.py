"""
Main dataset building module that orchestrates the parsing process.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from . import config
from .file_discovery import find_pairs, get_dataset_statistics
from .xml_parser import parse_xml_events, validate_events, get_event_statistics
from .labeling import derive_labels, get_label_distribution
from .audio_snippets import extract_and_save_snippet


def build(
    data_root: str,
    output_dir: str,
    export_snippets: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Build the IDMT-SMT-GUITAR dataset from raw files.
    
    Args:
        data_root: Root directory containing WAV and XML files
        output_dir: Output directory for processed dataset
        export_snippets: Whether to extract and save audio snippets
        verbose: Whether to print progress information
        
    Returns:
        DataFrame containing all metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("IDMT-SMT-GUITAR Dataset Parser")
        print("=" * 60)
        print(f"Data root: {data_root}")
        print(f"Output dir: {output_dir}")
        print(f"Export snippets: {export_snippets}")
        print()
    
    # Step 1: Discover WAV/XML pairs
    if verbose:
        print("Step 1: Discovering file pairs...")
    
    pairs = find_pairs(data_root)
    
    if not pairs:
        raise ValueError(f"No WAV/XML pairs found in {data_root}")
    
    if verbose:
        stats = get_dataset_statistics(pairs)
        print(f"  Found {stats['total_pairs']} file pairs")
        print(f"  Subsets: {stats['subsets']}")
        print()
    
    # Step 2: Parse XML annotations and build metadata
    if verbose:
        print("Step 2: Parsing annotations and building metadata...")
    
    all_records = []
    snippet_count = 0
    
    for wav_path, xml_path, subset_id in tqdm(pairs, disable=not verbose, desc="Processing files"):
        # Parse XML events
        events = parse_xml_events(xml_path)
        
        # Validate events
        events = validate_events(events, min_duration=config.MIN_DURATION_SEC)
        
        # Process each event
        for event_idx, event in enumerate(events):
            # Derive labels
            event = derive_labels(event)
            
            # Extract snippet if requested
            snippet_path = None
            if export_snippets:
                snippet_path = extract_and_save_snippet(
                    wav_path,
                    event,
                    output_path / "snippets",
                    subset_id,
                    event_idx
                )
                if snippet_path:
                    snippet_count += 1
            
            # Build metadata record
            record = {
                "subset_id": subset_id,
                "source_audio": str(wav_path),
                "source_annotation": str(xml_path),
                "event_index": event_idx,
                "onset_sec": event["onset_sec"],
                "offset_sec": event["offset_sec"],
                "duration_sec": event["duration_sec"],
                "pitch_midi": event["pitch_midi"],
                "string_number": event.get("string_number"),
                "fret_number": event.get("fret_number"),
                "excitation_style": event.get("excitation_style"),
                "expression_style": event.get("expression_style"),
                "label_category": event["label_category"],
                "is_harmonic": event["is_harmonic"],
                "is_dead_note": event["is_dead_note"],
                "is_general_note": event["is_general_note"],
            }
            
            if snippet_path:
                record["snippet_path"] = str(snippet_path)
            
            all_records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    if verbose:
        print(f"  Processed {len(all_records)} events")
        if export_snippets:
            print(f"  Exported {snippet_count} audio snippets")
        print()
    
    # Step 3: Save metadata
    if verbose:
        print("Step 3: Saving metadata...")
    
    # Save as CSV
    csv_path = output_path / config.METADATA_FILENAME
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"  Saved CSV: {csv_path}")
    
    # Save as Parquet (more efficient for large datasets)
    try:
        parquet_path = output_path / config.METADATA_PARQUET
        df.to_parquet(parquet_path, index=False)
        if verbose:
            print(f"  Saved Parquet: {parquet_path}")
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not save Parquet format: {e}")
    
    # Step 4: Display statistics
    if verbose:
        print()
        print("=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Total events: {len(df)}")
        print()
        print("Label distribution:")
        label_dist = df["label_category"].value_counts().to_dict()
        for label, count in sorted(label_dist.items()):
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        print()
        
        print("Expression styles:")
        expr_dist = df["expression_style"].value_counts().to_dict()
        for expr, count in sorted(expr_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {expr}: {count}")
        print()
        
        print(f"Duration range: {df['duration_sec'].min():.3f}s - {df['duration_sec'].max():.3f}s")
        print(f"Average duration: {df['duration_sec'].mean():.3f}s")
        print()
        print("=" * 60)
    
    return df


def sanity_check(df: pd.DataFrame) -> dict:
    """
    Run sanity checks on the parsed dataset.
    
    Args:
        df: DataFrame with parsed metadata
        
    Returns:
        Dictionary with check results
    """
    checks = {}
    
    # Check for missing values
    checks["missing_values"] = df.isnull().sum().to_dict()
    
    # Check for duplicate events
    duplicate_cols = ["source_audio", "onset_sec", "offset_sec"]
    checks["duplicate_events"] = df.duplicated(subset=duplicate_cols).sum()
    
    # Check for anomalous durations
    checks["very_short_events"] = (df["duration_sec"] < 0.05).sum()
    checks["very_long_events"] = (df["duration_sec"] > 10.0).sum()
    
    # Check label distribution
    checks["label_distribution"] = df["label_category"].value_counts().to_dict()
    
    return checks
