"""
Copy harmonic recordings from harmonics/ to note_clips/harmonic/

This script finds all WAV files in the harmonics/ directory structure
and copies them to note_clips/harmonic/ where they can be added to
the metadata for training.
"""

import shutil
from pathlib import Path
from collections import defaultdict

# Directories
HARMONICS_SOURCE = Path('harmonics')
NOTE_CLIPS_DEST = Path('note_clips/harmonic')

def main():
    print("=" * 70)
    print("  COPY HARMONICS TO NOTE_CLIPS")
    print("=" * 70)
    
    # Create destination directory if it doesn't exist
    NOTE_CLIPS_DEST.mkdir(parents=True, exist_ok=True)
    print(f"\nSource:      {HARMONICS_SOURCE}")
    print(f"Destination: {NOTE_CLIPS_DEST}")
    
    # Find all WAV files in harmonics directory
    print("\nScanning for WAV files...")
    wav_files = list(HARMONICS_SOURCE.rglob('*.wav'))
    print(f"Found {len(wav_files)} WAV files")
    
    if len(wav_files) == 0:
        print("\n✗ No WAV files found in harmonics directory!")
        return
    
    # Group by filename to detect duplicates
    filename_counts = defaultdict(list)
    for wav_file in wav_files:
        filename_counts[wav_file.name].append(wav_file)
    
    # Check for duplicates
    duplicates = {name: paths for name, paths in filename_counts.items() if len(paths) > 1}
    if duplicates:
        print(f"\n⚠ Warning: Found {len(duplicates)} filenames with multiple copies:")
        for name, paths in list(duplicates.items())[:5]:
            print(f"  {name}:")
            for path in paths:
                print(f"    - {path}")
        if len(duplicates) > 5:
            print(f"  ... and {len(duplicates) - 5} more")
        print("\n  Will copy first occurrence of each duplicate.")
    
    # Copy files
    print("\nCopying files...")
    copied = 0
    skipped = 0
    already_exists = []
    
    for filename, source_paths in filename_counts.items():
        source_path = source_paths[0]  # Use first occurrence
        dest_path = NOTE_CLIPS_DEST / filename
        
        if dest_path.exists():
            # Check if it's the same file
            if source_path.stat().st_size == dest_path.stat().st_size:
                already_exists.append(filename)
                skipped += 1
                continue
        
        # Copy file
        try:
            shutil.copy2(source_path, dest_path)
            copied += 1
            if copied <= 10:  # Show first 10
                print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ Error copying {filename}: {e}")
            skipped += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Total files found:     {len(wav_files)}")
    print(f"  Unique filenames:      {len(filename_counts)}")
    print(f"  Copied:                {copied}")
    print(f"  Skipped (exists):      {len(already_exists)}")
    print(f"  Errors:                {skipped - len(already_exists)}")
    
    if already_exists and len(already_exists) <= 10:
        print(f"\n  Files already in destination:")
        for name in already_exists:
            print(f"    - {name}")
    elif already_exists:
        print(f"\n  {len(already_exists)} files already exist in destination (not shown)")
    
    print("\n  Files are now in: {NOTE_CLIPS_DEST}")
    print("\n  Next step:")
    print("    python add_new_harmonics_to_metadata.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
