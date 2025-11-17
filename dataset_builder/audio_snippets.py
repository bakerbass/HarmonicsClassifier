"""
Audio snippet extraction module for isolating individual note events.
"""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Tuple, Optional
from . import config


def extract_snippet(
    audio_path: Path,
    onset: float,
    offset: float,
    pad: float = config.SNIPPET_PADDING_SEC
) -> Tuple[np.ndarray, int]:
    """
    Extract audio snippet for a single note event.
    
    Args:
        audio_path: Path to source audio file
        onset: Start time in seconds
        offset: End time in seconds
        pad: Padding to add before/after in seconds
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Load audio with original sample rate
    audio, sr = librosa.load(audio_path, sr=config.DEFAULT_SAMPLE_RATE, mono=True)
    
    # Calculate sample indices with padding
    start = max(0, int((onset - pad) * sr))
    end = min(len(audio), int((offset + pad) * sr))
    
    # Extract snippet
    snippet = audio[start:end]
    
    return snippet, sr


def save_snippet(
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: Path
) -> bool:
    """
    Save audio snippet to WAV file.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        output_path: Destination file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write WAV file
        sf.write(output_path, audio_data, sample_rate)
        return True
    except Exception as e:
        print(f"Error saving snippet to {output_path}: {e}")
        return False


def generate_snippet_filename(
    subset_id: str,
    basename: str,
    event_index: int,
    pitch: int,
    expression: Optional[str]
) -> str:
    """
    Generate standardized filename for audio snippet.
    
    Args:
        subset_id: Dataset subset identifier
        basename: Base name from source audio file
        event_index: Index of event within file
        pitch: MIDI pitch number
        expression: Expression style code
        
    Returns:
        Generated filename string
    """
    expr_str = expression if expression else "unknown"
    
    return config.SNIPPET_NAME_TEMPLATE.format(
        subset_id=subset_id,
        basename=basename,
        index=event_index,
        pitch=pitch,
        expr=expr_str
    )


def extract_and_save_snippet(
    audio_path: Path,
    event: dict,
    output_dir: Path,
    subset_id: str,
    event_index: int
) -> Optional[Path]:
    """
    Extract and save a single audio snippet.
    
    Args:
        audio_path: Source audio file path
        event: Event dictionary with onset/offset times
        output_dir: Output directory for snippets
        subset_id: Dataset subset identifier
        event_index: Index of event within file
        
    Returns:
        Path to saved snippet file, or None if failed
    """
    try:
        # Extract snippet
        audio_data, sr = extract_snippet(
            audio_path,
            event["onset_sec"],
            event["offset_sec"]
        )
        
        # Generate filename
        filename = generate_snippet_filename(
            subset_id,
            audio_path.stem,
            event_index,
            event["pitch_midi"],
            event.get("expression_style")
        )
        
        # Determine subdirectory based on label
        label_category = event.get("label_category", "unknown")
        snippet_path = output_dir / label_category / filename
        
        # Save snippet
        if save_snippet(audio_data, sr, snippet_path):
            return snippet_path
        else:
            return None
            
    except Exception as e:
        print(f"Error processing event {event_index} from {audio_path}: {e}")
        return None
