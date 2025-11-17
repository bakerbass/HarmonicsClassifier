"""
File discovery module for finding WAV/XML pairs in the IDMT-SMT-GUITAR dataset.
"""

import re
from pathlib import Path
from typing import List, Tuple


def find_pairs(root: str) -> List[Tuple[Path, Path, str]]:
    """
    Walk the root directory and find matching WAV/XML file pairs.
    
    Args:
        root: Root directory path containing the dataset
        
    Returns:
        List of tuples: (wav_path, xml_path, subset_id)
    """
    audio_files = {}
    xml_files = {}
    
    root_path = Path(root)
    
    # Collect all WAV and XML files
    for path in root_path.rglob("*"):
        if path.suffix.lower() == ".wav":
            audio_files[path.stem] = path
        elif path.suffix.lower() == ".xml":
            xml_files[path.stem] = path
    
    # Match pairs and extract subset information
    pairs = []
    for stem, wav_path in audio_files.items():
        if stem in xml_files:
            xml_path = xml_files[stem]
            subset_id = _extract_subset_id(wav_path)
            pairs.append((wav_path, xml_path, subset_id))
    
    return pairs


def _extract_subset_id(file_path: Path) -> str:
    """
    Extract subset ID from file path or directory structure.
    
    Args:
        file_path: Path to the audio/annotation file
        
    Returns:
        Subset identifier string (e.g., "dataset1", "dataset2")
    """
    # Check directory names for dataset indicators
    parts = file_path.parts
    
    for part in parts:
        # Look for patterns like "dataset1", "dataset_1", "subset1", etc.
        match = re.search(r'(dataset|subset|data)[-_]?(\d+)', part.lower())
        if match:
            return f"dataset{match.group(2)}"
    
    # Fallback: use parent directory name
    return file_path.parent.name


def get_dataset_statistics(pairs: List[Tuple[Path, Path, str]]) -> dict:
    """
    Generate statistics about discovered file pairs.
    
    Args:
        pairs: List of (wav_path, xml_path, subset_id) tuples
        
    Returns:
        Dictionary with dataset statistics
    """
    subsets = {}
    for wav_path, xml_path, subset_id in pairs:
        if subset_id not in subsets:
            subsets[subset_id] = 0
        subsets[subset_id] += 1
    
    return {
        "total_pairs": len(pairs),
        "subsets": subsets,
    }
