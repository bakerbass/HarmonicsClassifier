"""
Feature extraction module for SVM-based harmonics classification.

Extracts traditional machine learning features from audio files:
- MFCCs (Mel-Frequency Cepstral Coefficients)
- Spectral features (centroid, rolloff, contrast, bandwidth)
- Zero-crossing rate
- RMS energy
- Chroma features

These features are more interpretable and efficient than raw spectrograms
for traditional ML classifiers like SVMs.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Tuple, Optional


class AudioFeatureExtractor:
    """Extract acoustic features for machine learning classification."""
    
    def __init__(
        self,
        sr: int = 22050,
        duration: float = 3.0,
        n_mfcc: int = 20,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_chroma: int = 12,
        normalize_audio: bool = True,
        target_rms: float = 0.1,
    ):
        """
        Initialize feature extractor.
        
        Args:
            sr: Target sample rate
            duration: Maximum audio duration to process
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_chroma: Number of chroma bins
            normalize_audio: Whether to normalize audio amplitude
            target_rms: Target RMS level for normalization
        """
        self.sr = sr
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.normalize_audio = normalize_audio
        self.target_rms = target_rms
        
    def load_audio(
        self,
        audio_path: str,
        onset_sec: float = 0.0,
        duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        """
        Load audio file with specified offset and duration.
        
        Args:
            audio_path: Path to audio file
            onset_sec: Start time in seconds
            duration_sec: Duration to load (None = use self.duration)
            
        Returns:
            Audio signal as numpy array
        """
        if duration_sec is None:
            duration_sec = self.duration
            
        audio, _ = librosa.load(
            audio_path,
            sr=self.sr,
            offset=onset_sec,
            duration=duration_sec,
        )
        
        # Normalize amplitude to consistent RMS level
        if self.normalize_audio and len(audio) > 0:
            current_rms = np.sqrt(np.mean(audio ** 2))
            if current_rms > 1e-6:  # Avoid division by zero
                audio = audio * (self.target_rms / current_rms)
        
        # Pad or trim to fixed length
        target_length = int(self.sr * self.duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
            
        return audio
    
    def extract_mfcc_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract MFCC-based features.
        
        Returns statistical summaries (mean, std, min, max) of each MFCC coefficient.
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        features = {}
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_min'] = np.min(mfccs[i])
            features[f'mfcc_{i}_max'] = np.max(mfccs[i])
            
        return features
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features."""
        # Spectral centroid - "center of mass" of spectrum
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral rolloff - frequency below which X% of energy is contained
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Spectral contrast - difference between peaks and valleys
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        
        # Zero-crossing rate - relevant for percussive vs harmonic content
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        
        features = {
            'spectral_centroid_mean': np.mean(centroid),
            'spectral_centroid_std': np.std(centroid),
            'spectral_centroid_min': np.min(centroid),
            'spectral_centroid_max': np.max(centroid),
            
            'spectral_rolloff_mean': np.mean(rolloff),
            'spectral_rolloff_std': np.std(rolloff),
            'spectral_rolloff_min': np.min(rolloff),
            'spectral_rolloff_max': np.max(rolloff),
            
            'spectral_bandwidth_mean': np.mean(bandwidth),
            'spectral_bandwidth_std': np.std(bandwidth),
            'spectral_bandwidth_min': np.min(bandwidth),
            'spectral_bandwidth_max': np.max(bandwidth),
            
            'zero_crossing_rate_mean': np.mean(zcr),
            'zero_crossing_rate_std': np.std(zcr),
            'zero_crossing_rate_min': np.min(zcr),
            'zero_crossing_rate_max': np.max(zcr),
        }
        
        # Add spectral contrast bands (7 bands by default)
        for i in range(contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(contrast[i])
            features[f'spectral_contrast_{i}_std'] = np.std(contrast[i])
            
        return features
    
    def extract_chroma_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract chroma (pitch class) features."""
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
        )
        
        features = {}
        for i in range(self.n_chroma):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i}_std'] = np.std(chroma[i])
            
        return features
    
    def extract_energy_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract energy-related features."""
        # RMS energy
        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Onset strength (for detecting note attacks)
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        
        features = {
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
            'rms_min': np.min(rms),
            'rms_max': np.max(rms),
            
            'onset_strength_mean': np.mean(onset_env),
            'onset_strength_std': np.std(onset_env),
            'onset_strength_max': np.max(onset_env),
        }
        
        return features
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract temporal envelope features."""
        # Compute amplitude envelope
        envelope = np.abs(audio)
        
        # Attack time (time to reach 90% of peak)
        peak = np.max(envelope)
        threshold = 0.9 * peak
        attack_idx = np.where(envelope >= threshold)[0]
        attack_time = attack_idx[0] / self.sr if len(attack_idx) > 0 else 0.0
        
        # Decay characteristics (how quickly energy drops)
        if len(envelope) > 0:
            # Divide into quarters and measure energy decay
            quarter = len(envelope) // 4
            energy_q1 = np.mean(envelope[:quarter] ** 2)
            energy_q2 = np.mean(envelope[quarter:2*quarter] ** 2)
            energy_q3 = np.mean(envelope[2*quarter:3*quarter] ** 2)
            energy_q4 = np.mean(envelope[3*quarter:] ** 2)
            
            # Decay rate (normalized energy change)
            decay_rate = (energy_q1 - energy_q4) / (energy_q1 + 1e-8)
        else:
            energy_q1 = energy_q2 = energy_q3 = energy_q4 = 0.0
            decay_rate = 0.0
        
        features = {
            'attack_time': attack_time,
            'energy_q1': energy_q1,
            'energy_q2': energy_q2,
            'energy_q3': energy_q3,
            'energy_q4': energy_q4,
            'decay_rate': decay_rate,
        }
        
        return features
    
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract all features from audio signal.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary of feature_name: value
        """
        # Normalize amplitude if enabled
        if self.normalize_audio and len(audio) > 0:
            current_rms = np.sqrt(np.mean(audio ** 2))
            if current_rms > 1e-6:  # Avoid division by zero
                audio = audio * (self.target_rms / current_rms)
        
        features = {}
        
        # MFCC features (20 coefficients × 4 stats = 80 features)
        features.update(self.extract_mfcc_features(audio))
        
        # Spectral features (~30 features)
        features.update(self.extract_spectral_features(audio))
        
        # Chroma features (12 × 2 stats = 24 features)
        features.update(self.extract_chroma_features(audio))
        
        # Energy features (~7 features)
        features.update(self.extract_energy_features(audio))
        
        # Temporal features (~6 features)
        features.update(self.extract_temporal_features(audio))
        
        return features
    
    def extract_from_file(
        self,
        audio_path: str,
        onset_sec: float = 0.0,
        duration_sec: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Load audio file and extract all features.
        
        Args:
            audio_path: Path to audio file
            onset_sec: Start time in seconds
            duration_sec: Duration to load
            
        Returns:
            Dictionary of features
        """
        audio = self.load_audio(audio_path, onset_sec, duration_sec)
        return self.extract_all_features(audio)
    
    def get_feature_names(self) -> list:
        """
        Get ordered list of feature names.
        
        Useful for creating consistent feature matrices.
        """
        # Extract features from a random signal to get feature names
        dummy_audio = np.random.randn(int(self.sr * self.duration))
        features = self.extract_all_features(dummy_audio)
        return sorted(features.keys())


def extract_features_from_metadata(
    metadata_df,
    extractor: Optional[AudioFeatureExtractor] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Extract features for all samples in a metadata DataFrame.
    
    Args:
        metadata_df: DataFrame with columns: source_audio, onset_sec, 
                     duration_sec (optional), label_category
        extractor: AudioFeatureExtractor instance (creates default if None)
        verbose: Show progress bar
        
    Returns:
        Tuple of (features_matrix, labels_array, feature_names)
    """
    if extractor is None:
        extractor = AudioFeatureExtractor()
    
    feature_names = extractor.get_feature_names()
    n_features = len(feature_names)
    n_samples = len(metadata_df)
    
    # Initialize output arrays
    X = np.zeros((n_samples, n_features))
    label_map = {'harmonic': 0, 'dead_note': 1, 'general_note': 2}
    y = np.array([label_map[label] for label in metadata_df['label_category']])
    
    # Extract features
    iterator = metadata_df.iterrows()
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(iterator, total=n_samples, desc="Extracting features")
    
    for i, (_, row) in enumerate(iterator):
        try:
            duration = row.get('duration_sec', None)
            features = extractor.extract_from_file(
                row['source_audio'],
                onset_sec=row['onset_sec'],
                duration_sec=duration,
            )
            
            # Fill feature vector in consistent order
            for j, feat_name in enumerate(feature_names):
                X[i, j] = features[feat_name]
                
        except Exception as e:
            if verbose:
                print(f"\nWarning: Failed to extract features for row {i}: {e}")
            # Leave as zeros
            
    return X, y, feature_names


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    
    print("AudioFeatureExtractor Test")
    print("=" * 60)
    
    extractor = AudioFeatureExtractor(sr=22050, duration=3.0)
    
    # Test with synthetic signal
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create a harmonic signal (fundamental + harmonics)
    f0 = 220  # A3
    signal = (np.sin(2 * np.pi * f0 * t) +
              0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
              0.25 * np.sin(2 * np.pi * 3 * f0 * t))
    
    features = extractor.extract_all_features(signal)
    
    print(f"\nExtracted {len(features)} features:")
    print("\nSample features:")
    for i, (name, value) in enumerate(list(features.items())[:10]):
        print(f"  {name:30s}: {value:.4f}")
    print("  ...")
    
    print(f"\nFeature groups:")
    mfcc_count = sum(1 for k in features if k.startswith('mfcc'))
    spectral_count = sum(1 for k in features if k.startswith('spectral'))
    chroma_count = sum(1 for k in features if k.startswith('chroma'))
    rms_count = sum(1 for k in features if k.startswith('rms'))
    
    print(f"  MFCC features:     {mfcc_count}")
    print(f"  Spectral features: {spectral_count}")
    print(f"  Chroma features:   {chroma_count}")
    print(f"  Energy features:   {rms_count}")
    print(f"  Other features:    {len(features) - mfcc_count - spectral_count - chroma_count - rms_count}")
    
    print("\n" + "=" * 60)
    print("Feature extraction module ready for SVM training!")
