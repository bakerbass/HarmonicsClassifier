"""
Main entry point for running the IDMT-SMT-GUITAR dataset parser.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_builder.build_dataset import build, sanity_check


def main():
    """Main function to parse command-line arguments and build dataset."""
    parser = argparse.ArgumentParser(
        description="Parse IDMT-SMT-GUITAR dataset and generate ML-ready metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse dataset and create metadata only
  python run_build_dataset.py --data-root ./IDMT-SMT-GUITAR --output-dir ./processed_dataset

  # Parse dataset and export audio snippets
  python run_build_dataset.py --data-root ./IDMT-SMT-GUITAR --output-dir ./processed_dataset --export-snippets

  # Run in quiet mode
  python run_build_dataset.py --data-root ./IDMT-SMT-GUITAR --output-dir ./processed_dataset --quiet
        """
    )
    
    parser.add_argument(
        "--data-root",
        default="./IDMT-SMT-GUITAR_V2",
        help="Root directory containing IDMT-SMT-GUITAR dataset (WAV and XML files)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="./processed_dataset",
        help="Output directory for processed dataset and metadata"
    )
    
    parser.add_argument(
        "--export-snippets",
        action="store_true",
        help="Extract and save individual audio snippets for each note event"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run additional sanity checks on the parsed dataset"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: Data root directory does not exist: {data_root}")
        sys.exit(1)
    
    # Build dataset
    try:
        df = build(
            data_root=str(data_root),
            output_dir=args.output_dir,
            export_snippets=args.export_snippets,
            verbose=not args.quiet
        )
        
        # Run sanity checks if requested
        if args.sanity_check:
            if not args.quiet:
                print("\nRunning sanity checks...")
            
            checks = sanity_check(df)
            
            if not args.quiet:
                print("\nSanity Check Results:")
                print("=" * 60)
                
                print("\nMissing values:")
                for col, count in checks["missing_values"].items():
                    if count > 0:
                        print(f"  {col}: {count}")
                
                print(f"\nDuplicate events: {checks['duplicate_events']}")
                print(f"Very short events (<50ms): {checks['very_short_events']}")
                print(f"Very long events (>10s): {checks['very_long_events']}")
                print()
        
        if not args.quiet:
            print("\n✓ Dataset parsing completed successfully!")
            print(f"  Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"\n✗ Error during dataset parsing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
