# Agentic Planning: Guitar Harmonics Dataset Creation Tool

## Project Overview
Create a PyQt6-based desktop application for recording and annotating guitar notes to build a labeled dataset for a binary classification model (harmonic vs. non-harmonic).

## Project Goals
1. Enable efficient recording of guitar notes using sounddevice
2. Provide real-time audio visualization for quality control
3. Allow manual annotation (harmonic/non-harmonic) of recorded samples
4. Organize and save annotated audio data for ML model training
5. Support batch recording sessions with metadata tracking

## Technical Stack
- **GUI Framework**: PyQt6
- **Audio Recording**: sounddevice + numpy
- **Audio Processing**: scipy, librosa (optional for visualization)
- **Data Storage**: Structured filesystem + CSV/JSON metadata
- **Python Version**: 3.8+

## Architecture Components

### 1. Audio Recording Module
- **Functionality**:
  - Capture audio from default input device (microphone/audio interface)
  - Configurable sample rate (default: 44100 Hz)
  - Configurable recording duration (e.g., 2-3 seconds per note)
  - Real-time level monitoring
  - Support for manual start/stop or auto-triggered recording
  
- **Key Classes**:
  - `AudioRecorder`: Handles sounddevice recording operations
  - `AudioBuffer`: Manages recorded audio data

### 2. GUI Application Module
- **Main Window Components**:
  - Recording controls (Start/Stop button, duration selector)
  - Real-time waveform display
  - Annotation buttons (Harmonic/Non-Harmonic/Discard)
  - Session metadata inputs (string number, fret position, notes)
  - Sample counter and session progress display
  - Audio playback controls for review
  
- **Key Classes**:
  - `MainWindow`: Primary application window
  - `WaveformWidget`: Custom widget for audio visualization
  - `AnnotationPanel`: Controls for labeling samples
  - `SessionManager`: Tracks recording session state

### 3. Data Management Module
- **Functionality**:
  - Save audio files in organized directory structure
  - Generate and maintain metadata CSV/JSON
  - Support for session-based organization
  - Data versioning and backup capabilities
  
- **Directory Structure**:
  ```
  dataset/
  ├── harmonics/
  │   ├── session_001/
  │   │   ├── harmonic_001.wav
  │   │   ├── harmonic_002.wav
  │   │   └── ...
  │   └── session_002/
  ├── non_harmonics/
  │   ├── session_001/
  │   │   ├── non_harmonic_001.wav
  │   │   └── ...
  │   └── session_002/
  └── metadata.csv
  ```
  
- **Metadata Fields**:
  - filename
  - label (harmonic/non_harmonic)
  - session_id
  - timestamp
  - duration
  - sample_rate
  - string_number (optional)
  - fret_position (optional)
  - notes (optional)

### 4. Audio Visualization Module
- **Functionality**:
  - Real-time waveform plotting
  - Optional spectrogram display
  - Peak detection and level indicators
  
- **Key Libraries**:
  - PyQtGraph for fast plotting
  - matplotlib (alternative) for more advanced visualizations

## Implementation Phases

### Phase 1: Core Recording Functionality
- [ ] Set up project structure and dependencies
- [ ] Implement AudioRecorder class with sounddevice
- [ ] Create basic PyQt6 window with recording button
- [ ] Test audio capture and save to WAV file
- [ ] Implement basic error handling for audio device issues

### Phase 2: GUI Development
- [ ] Design main window layout with Qt Designer or code
- [ ] Add recording controls (start/stop, duration selector)
- [ ] Implement annotation buttons (Harmonic/Non-Harmonic/Discard)
- [ ] Add session metadata input fields
- [ ] Create sample counter and progress indicators
- [ ] Implement audio playback for review

### Phase 3: Visualization
- [ ] Integrate PyQtGraph for waveform display
- [ ] Implement real-time waveform updates during recording
- [ ] Add level meter for input monitoring
- [ ] Optional: Add spectrogram view

### Phase 4: Data Management
- [ ] Implement directory structure creation
- [ ] Create file naming and saving logic
- [ ] Implement metadata CSV generation and updates
- [ ] Add session management (new session, load session)
- [ ] Implement data export/backup functionality

### Phase 5: Polish & Features
- [ ] Add keyboard shortcuts for faster annotation
- [ ] Implement undo/redo for annotations
- [ ] Add audio preprocessing (normalization, trimming silence)
- [ ] Create settings dialog (audio device selection, paths, etc.)
- [ ] Add dataset statistics viewer
- [ ] Implement data validation and quality checks

### Phase 6: Testing & Documentation
- [ ] Test on different audio interfaces
- [ ] Create user documentation
- [ ] Add code documentation and comments
- [ ] Create example workflow tutorial
- [ ] Test edge cases and error scenarios

## Key Features Priority

### Must-Have (MVP)
1. Record audio clips (fixed duration)
2. Label clips as harmonic/non-harmonic
3. Save labeled data to organized folders
4. Basic waveform visualization
5. Session-based organization

### Should-Have
1. Audio playback for review before labeling
2. Real-time level monitoring
3. Metadata tracking (string, fret, notes)
4. Sample counter and progress tracking
5. Keyboard shortcuts

### Nice-to-Have
1. Spectrogram visualization
2. Auto-triggered recording (threshold-based)
3. Batch relabeling tools
4. Dataset statistics and analytics
5. Audio preprocessing options
6. Multi-session project management

## Technical Considerations

### Audio Configuration
- **Sample Rate**: 44100 Hz (standard for music)
- **Bit Depth**: 16-bit or 24-bit
- **Channels**: Mono (single guitar input)
- **Recording Duration**: 2-3 seconds (adjustable)
- **Buffer Size**: Configure for low latency

### File Format
- **Format**: WAV (uncompressed, lossless)
- **Alternative**: FLAC for compression if storage is concern

### Performance Optimization
- Use separate thread for audio recording to prevent GUI blocking
- Implement buffering for smooth waveform updates
- Optimize file I/O operations

### Error Handling
- Handle missing/disconnected audio devices
- Validate file write permissions
- Handle disk space issues
- Provide user-friendly error messages

## Development Workflow

1. **Setup**: Create virtual environment, install dependencies
2. **Iterative Development**: Build and test each phase incrementally
3. **User Testing**: Test workflow with actual guitar recording
4. **Refinement**: Adjust UI/UX based on real usage
5. **Documentation**: Document usage and code

## Dependencies

```
PyQt6
sounddevice
numpy
scipy
pyqtgraph
```

Optional:
```
librosa (for advanced audio analysis)
matplotlib (alternative visualization)
pandas (for metadata management)
```

## Success Metrics
- Ability to record 100+ samples per session efficiently
- Less than 5 seconds per sample (record + annotate)
- Reliable audio capture without dropouts
- Organized, ready-to-use dataset for ML training
- Intuitive UI requiring minimal learning curve

## Next Steps
1. Set up project structure
2. Create requirements.txt
3. Implement basic AudioRecorder class
4. Build minimal GUI with recording button
5. Test end-to-end: record → save → verify file

---

**Document Version**: 1.0  
**Last Updated**: November 17, 2025  
**Status**: Planning Phase
