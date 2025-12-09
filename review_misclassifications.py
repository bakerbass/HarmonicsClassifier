"""
Review harmonic misclassifications from training results.

This script helps you listen to and analyze false positives and false negatives
for harmonic classifications.

Usage:
    python review_misclassifications.py --results models/
    python review_misclassifications.py --results models/ --type false_positives
    python review_misclassifications.py --results models/ --type false_negatives --limit 10
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import subprocess
import sys


def load_misclassifications(results_dir):
    """Load misclassification data from results directory."""
    misclass_path = results_dir / 'harmonic_misclassifications.json'
    
    if not misclass_path.exists():
        print(f"Error: Misclassifications file not found: {misclass_path}")
        print("Make sure you've run training with the updated script.")
        return None
    
    with open(misclass_path, 'r') as f:
        data = json.load(f)
    
    return data


def print_summary(data):
    """Print summary of misclassifications."""
    n_fp = len(data['false_positives'])
    n_fn = len(data['false_negatives'])
    
    print("="*70)
    print("HARMONIC MISCLASSIFICATION SUMMARY")
    print("="*70)
    print(f"False Positives (predicted harmonic, actually not): {n_fp}")
    print(f"False Negatives (predicted not harmonic, actually harmonic): {n_fn}")
    print(f"Total harmonic misclassifications: {n_fp + n_fn}")
    print("="*70)


def display_misclassifications(items, title):
    """Display misclassifications in a formatted table."""
    if len(items) == 0:
        print(f"\n✓ No {title.lower()} found!")
        return
    
    print(f"\n{title}:")
    print("-"*70)
    
    for i, item in enumerate(items, 1):
        print(f"\n{i}. Audio: {item['audio_file']}")
        print(f"   Time: {item['onset_sec']:.2f}s (duration: {item['duration_sec']:.2f}s)")
        print(f"   True label: {item['true_label']}")
        print(f"   Predicted: {item['predicted_label']}")
        if item['pitch_midi']:
            print(f"   Pitch: MIDI {item['pitch_midi']}")


def play_audio(audio_path, onset, duration):
    """Play audio segment using system default player."""
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        return False
    
    # Try to play with ffplay (from ffmpeg) if available
    try:
        # ffplay with seek and duration
        cmd = ['ffplay', '-nodisp', '-autoexit', '-ss', str(onset), '-t', str(duration), str(audio_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Fallback: just open the file with default player
    try:
        if sys.platform == 'win32':
            subprocess.run(['start', '', str(audio_path)], shell=True)
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(audio_path)])
        else:
            subprocess.run(['xdg-open', str(audio_path)])
        print(f"Opened {audio_path} with default player (couldn't seek to {onset}s)")
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False


def interactive_review(items, title):
    """Interactive review of misclassifications with audio playback."""
    if len(items) == 0:
        print(f"\n✓ No {title.lower()} found!")
        return
    
    print(f"\n{title} - Interactive Review")
    print("="*70)
    print("Commands: [n]ext, [p]lay, [r]eplay, [i]nfo, [q]uit")
    print("="*70)
    
    i = 0
    while i < len(items):
        item = items[i]
        print(f"\n[{i+1}/{len(items)}] {item['audio_file']}")
        print(f"  True: {item['true_label']} | Predicted: {item['predicted_label']}")
        print(f"  Time: {item['onset_sec']:.2f}s - {item['onset_sec'] + item['duration_sec']:.2f}s")
        
        while True:
            cmd = input("\nCommand [n/p/r/i/q]: ").strip().lower()
            
            if cmd == 'n':
                i += 1
                break
            elif cmd == 'p' or cmd == 'r':
                print("Playing audio...")
                play_audio(item['audio_file'], item['onset_sec'], item['duration_sec'])
            elif cmd == 'i':
                print(f"\nDetailed info:")
                for key, value in item.items():
                    print(f"  {key}: {value}")
            elif cmd == 'q':
                return
            else:
                print("Invalid command. Use n/p/r/i/q")


def main():
    parser = argparse.ArgumentParser(description='Review harmonic misclassifications')
    parser.add_argument('--results', default='models/', help='Results directory')
    parser.add_argument('--type', choices=['false_positives', 'false_negatives', 'both'], 
                       default='both', help='Type of misclassifications to review')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of items to show')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode with audio playback')
    parser.add_argument('--export-csv', action='store_true', help='Export to CSV files')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Load misclassifications
    print("Loading misclassifications...")
    data = load_misclassifications(results_dir)
    if data is None:
        return
    
    # Print summary
    print_summary(data)
    
    # Get items to review
    if args.type == 'both':
        fp_items = data['false_positives'][:args.limit] if args.limit else data['false_positives']
        fn_items = data['false_negatives'][:args.limit] if args.limit else data['false_negatives']
    elif args.type == 'false_positives':
        fp_items = data['false_positives'][:args.limit] if args.limit else data['false_positives']
        fn_items = []
    else:  # false_negatives
        fp_items = []
        fn_items = data['false_negatives'][:args.limit] if args.limit else data['false_negatives']
    
    # Interactive or display mode
    if args.interactive:
        if fp_items:
            interactive_review(fp_items, "FALSE POSITIVES")
        if fn_items:
            interactive_review(fn_items, "FALSE NEGATIVES")
    else:
        if fp_items:
            display_misclassifications(fp_items, "FALSE POSITIVES")
        if fn_items:
            display_misclassifications(fn_items, "FALSE NEGATIVES")
    
    # Export CSV if requested
    if args.export_csv:
        if fp_items:
            fp_df = pd.DataFrame(fp_items)
            fp_path = results_dir / 'false_positives_harmonics.csv'
            fp_df.to_csv(fp_path, index=False)
            print(f"\n✓ Exported false positives to {fp_path}")
        
        if fn_items:
            fn_df = pd.DataFrame(fn_items)
            fn_path = results_dir / 'false_negatives_harmonics.csv'
            fn_df.to_csv(fn_path, index=False)
            print(f"✓ Exported false negatives to {fn_path}")


if __name__ == "__main__":
    main()
