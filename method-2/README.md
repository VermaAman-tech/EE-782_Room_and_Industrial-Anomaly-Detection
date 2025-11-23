# Multimodal Room Monitor - Comprehensive Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Project Structure & File Descriptions](#project-structure--file-descriptions)
4. [Dataflow: Input to Output](#dataflow-input-to-output)
5. [Libraries & Dependencies](#libraries--dependencies)
6. [Innovative Features & Ideas](#innovative-features--ideas)
7. [Installation & Setup](#installation--setup)
8. [Running the Project](#running-the-project)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

The **Multimodal Room Monitor** is an advanced real-time anomaly detection and event monitoring system that combines computer vision and audio analysis to provide comprehensive environmental awareness. The system processes synchronized video frames and audio streams to detect objects, track movements, identify audio events, and fuse multimodal information for intelligent anomaly detection.

### Key Capabilities
- **Real-time Visual Analysis**: Object detection, tracking, segmentation, and motion analysis
- **Advanced Audio Processing**: Multi-model audio classification using AST, Wav2Vec2, and HuBERT
- **Cross-Modal Fusion**: Transformer-based fusion of visual and audio features
- **Anomaly Detection**: Multiple anomaly scoring methods (statistical, autoencoder-based)
- **Interactive Dashboard**: Streamlit-based web interface for live monitoring
- **Industrial Monitoring**: Specialized modes for industrial safety and anomaly detection

---

## System Architecture

flowchart LR

  %% Sensors
  subgraph SENSORS [Input Layer]
    CAM["Camera (video frames)"]
    MIC["Microphone (audio stream)"]
  end

  %% Preprocessing
  subgraph PRE [Preprocessing Layer]
    IMG_PRE["Image Preprocessing: denoise, wavelets, flow"]
    AUD_PRE["Audio Preprocessing: resample, STFT, mel, CWT"]
  end

  %% Visual Analysis
  subgraph VIS [Visual Analysis]
    DET["YOLO / DETR Detector"]
    TRK["ByteTrack Tracker"]
    SEG["SAM Segmenter"]
    FLOW["Optical Flow and Motion Analysis"]
  end

  %% Audio Analysis
  subgraph AUD [Audio Analysis]
    AST["AST Model"]
    WV2["Wav2Vec2"]
    HUB["HuBERT"]
    SPEC["Spectral Features"]
  end

  %% Fusion
  subgraph FUS [Fusion Layer]
    PROJ["Projection and Positional Embeddings"]
    XATT["Cross Modal Transformer"]
    POOL["Temporal Pooling"]
    HEADS["Task Heads: Motion and Event Classifier"]
  end

  %% Output
  subgraph OUT [Output Layer]
    ANOM["Anomaly Scoring"]
    LOG["Event Logging"]
    DASH["Streamlit Dashboard"]
    ARTIFACTS["Saved Artifacts"]
  end

  %% Flow connections
  CAM --> IMG_PRE
  MIC --> AUD_PRE

  IMG_PRE --> DET
  DET --> TRK
  DET --> SEG
  IMG_PRE --> FLOW
  FLOW --> VIS_RESULTS["Visual Tokens"]

  AUD_PRE --> AST
  AUD_PRE --> WV2
  AUD_PRE --> HUB
  AUD_PRE --> SPEC
  AST --> AUD_EMB["Audio Embeddings"]
  WV2 --> AUD_EMB
  HUB --> AUD_EMB
  SPEC --> AUD_FEAT["Audio Tokens"]

  TRK --> VIS_RESULTS
  SEG --> VIS_RESULTS

  VIS_RESULTS --> PROJ
  AUD_EMB --> PROJ
  AUD_FEAT --> PROJ

  PROJ --> XATT --> POOL --> HEADS

  HEADS --> ANOM
  ANOM --> LOG
  LOG --> DASH
  DASH --> ARTIFACTS

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                             │
│  ┌──────────────┐              ┌──────────────┐                │
│  │   Camera     │              │  Microphone  │                │
│  │  (Video)     │              │   (Audio)    │                │
│  └──────┬───────┘              └──────┬───────┘                │
└─────────┼──────────────────────────────┼────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING LAYER                           │
│  ┌──────────────┐              ┌──────────────┐                │
│  │ Image        │              │  Audio        │                │
│  │ Preprocessing│              │  Preprocessing│                │
│  │ - Denoising  │              │  - Resampling │                │
│  │ - Wavelets   │              │  - Normalize  │                │
│  └──────┬───────┘              └──────┬───────┘                │
└─────────┼──────────────────────────────┼────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ANALYSIS LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  VISUAL ANALYSIS                                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │   │
│  │  │ YOLO     │  │ ByteTrack │  │ MobileSAM│  │ Optical │ │   │
│  │  │ Detector │→ │ Tracker   │→ │ Segmenter│  │ Flow    │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  AUDIO ANALYSIS                                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │   │
│  │  │ AST      │  │ Wav2Vec2 │  │ HuBERT   │  │ Spectral │ │   │
│  │  │ Model    │  │ Model    │  │ Model    │  │ Features │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────┬──────────────────────────────┬────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FUSION LAYER                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Cross-Modal Transformer                                 │   │
│  │  - Multi-Head Cross-Attention                           │   │
│  │  - Visual-Audio Alignment                               │   │
│  │  - Temporal Feature Fusion                              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────┬──────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Anomaly  │  │ Event    │  │ Motion   │  │ Dashboard│       │
│  │ Detection│  │ Logging  │  │ Analysis │  │ Display  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **Sensors** capture raw video frames and audio samples
2. **Preprocessing** modules clean and prepare data
3. **Models** perform specialized analysis (detection, classification, segmentation)
4. **Fusion** combines multimodal features
5. **Utils** handle synchronization, anomaly scoring, and configuration
6. **UI** displays results and provides interactive controls

---

## Project Structure & File Descriptions

### Root Directory
```
CourseProject/
├── configs/              # Configuration files
├── models/               # Core ML models and neural networks
├── preprocess/           # Data preprocessing utilities
├── sensors/              # Hardware interface modules
├── ui/                   # Streamlit user interface
├── utils/                # Helper utilities and tools
├── anomalies/            # Saved anomaly detections
│   ├── audio/           # Audio snippets from anomalies
│   └── frames/          # Image frames from anomalies
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Detailed File Descriptions

#### `configs/config.yaml`
**Purpose**: Central configuration file for all system parameters

**Key Settings**:
- `device`: Hardware device selection ('auto', 'cuda', 'cpu')
- `burst`: Frame capture settings (number of frames, interval)
- `audio`: Audio recording parameters (sample rate, duration, channels)
- `vision`: Visual analysis settings (detector type, confidence thresholds, tracker)
- `audio_model`: HuggingFace model identifier for audio classification
- `fusion`: Transformer fusion network hyperparameters
- `ui`: Interface display settings

**Usage**: Loaded at startup via `utils.config.load_config()`

---

#### `models/audio_model.py`
**Purpose**: Advanced multi-model audio analysis system

**Class: `AdvancedAudioAnalyzer`**

**Key Features**:
- **AST (Audio Spectrogram Transformer)**: General audio classification
- **Wav2Vec2**: Speech recognition and transcription
- **HuBERT**: Acoustic scene understanding
- **Embedding Fusion**: Combines features from all three models

**Methods**:
- `__init__()`: Initializes all three models and fusion layer
- `_prepare_audio()`: Resamples and normalizes audio to 16kHz
- `analyze()`: Comprehensive audio analysis returning:
  - Audio event classifications
  - Speech transcription
  - Fused acoustic embeddings
  - Top predictions with confidence scores
- `classify()`: Legacy compatibility method

**Innovation**: Multi-model ensemble approach for robust audio understanding

---

#### `models/detector.py`
**Purpose**: Dual-detector object detection system

**Class: `AdvancedDetector` (aliased as `FastDetector`)**

**Key Features**:
- **Primary Detector**: YOLOv8 (Ultralytics) - Fast, efficient
- **Secondary Detector**: DETR (Facebook) - More accurate for complex scenes
- **Non-Maximum Suppression**: Removes duplicate detections across models

**Methods**:
- `__init__()`: Initializes YOLO and optionally DETR
- `detect()`: Processes frames and returns bounding boxes with:
  - Coordinates (xyxy format)
  - Class IDs and labels
  - Confidence scores
  - Source detector (yolo/detr)
- `_nms()`: Custom NMS implementation for cross-detector deduplication

**Innovation**: Hybrid detection combining speed (YOLO) and accuracy (DETR)

---

#### `models/fusion.py`
**Purpose**: Cross-modal transformer for visual-audio fusion

**Class: `AdvancedCrossModalTransformer`**

**Architecture**:
```
Input Projections (Visual & Audio)
    ↓
Positional Embeddings
    ↓
Multi-Layer Cross-Attention:
    - Visual → Audio attention
    - Audio → Visual attention
    - Feed-forward networks
    - Layer normalization
    ↓
Sequence Pooling
    ↓
Task Heads:
    - Motion Classifier (binary)
    - Event Classifier (32 categories)
```

**Key Components**:
- `MultiHeadCrossAttention`: Bidirectional attention mechanism
- `AdvancedCrossModalTransformer`: Main fusion network
- `build_fusion_head()`: Factory function for creating fusion models

**Innovation**: Bidirectional cross-attention allows each modality to inform the other

---

#### `models/segmenter.py`
**Purpose**: Instance segmentation using SAM (Segment Anything Model)

**Class: `AdvancedSegmenter`**

**Supported Models**:
- **MobileSAM**: Lightweight, fast segmentation
- **FastSAM**: Alternative fast segmentation model
- **DummyBoxMasker**: Fallback for testing without model weights

**Methods**:
- `segment_boxes()`: Generates precise masks for detected bounding boxes
- `segment_full()`: Full-image segmentation without prompts
- `get_features()`: Extracts visual features using SAM's encoder

**Usage**: Converts bounding boxes to pixel-accurate masks for detailed analysis

---

#### `models/tracker.py`
**Purpose**: Object tracking across video frames

**Class: `ByteTrackWrapper`**

**Implementation**: Wraps Ultralytics YOLO's built-in ByteTrack tracker

**Features**:
- Multi-object tracking with persistent IDs
- Handles occlusions and re-identification
- Integrates seamlessly with YOLO detections

**Output**: Tracked objects with:
- Persistent track IDs
- Bounding boxes per frame
- Class and confidence information

---

#### `preprocess/image.py`
**Purpose**: Image preprocessing and feature extraction

**Functions**:
- `denoise_frame()`: Removes noise using Non-Local Means algorithm
- `compute_wavelet_energy()`: Extracts texture energy using wavelet decomposition
- `optical_flow_tvl1()`: Computes dense optical flow (motion vectors)
- `draw_flow_arrows()`: Visualizes optical flow as arrows

**Techniques**:
- **Wavelet Transform**: Multi-resolution texture analysis
- **TV-L1 Optical Flow**: Dense motion estimation
- **Denoising**: Improves detection accuracy

---

#### `preprocess/audio.py`
**Purpose**: Audio signal processing and feature extraction

**Functions**:
- `compute_stft()`: Short-Time Fourier Transform for time-frequency analysis
- `compute_mel_spectrogram()`: Mel-scale spectrogram (perceptual frequency scale)
- `compute_cwt_scalogram()`: Continuous Wavelet Transform for multi-scale analysis
- `spectral_features()`: Extracts statistical features:
  - Spectral centroid (brightness)
  - Spectral bandwidth (spread)
  - Spectral rolloff (high-frequency cutoff)
  - Zero-crossing rate (temporal characteristics)

**Techniques**:
- **STFT**: Standard frequency analysis
- **Mel Spectrogram**: Human-perception-aligned features
- **CWT**: Multi-resolution time-frequency analysis
- **Spectral Statistics**: Compact feature representation

---

#### `sensors/camera.py`
**Purpose**: Camera interface and frame capture

**Functions**:
- `list_cameras()`: Detects available cameras using multiple OpenCV backends
  - Supports DirectShow (Windows), MSMF, and generic backends
  - Verifies cameras can actually read frames
- `capture_burst()`: Captures multiple frames with timestamps
  - Configurable frame count and interval
  - Returns frames and precise timestamps for synchronization

**Platform Support**: Optimized for Windows with fallbacks for other OS

---

#### `sensors/mic.py`
**Purpose**: Microphone interface and audio recording

**Functions**:
- `list_microphones()`: Lists available audio input devices
- `record_audio()`: Records audio with specified parameters
  - Duration, sample rate, channel configuration
  - Returns numpy array of audio samples

**Implementation**: Uses `sounddevice` for cross-platform audio I/O

---

#### `utils/anomaly.py`
**Purpose**: Anomaly detection and scoring

**Functions**:
- `image_anomaly_score()`: Statistical anomaly detection using z-scores
- `audio_anomaly_score()`: Audio anomaly based on energy and spectral centroid
- `ae_anomaly_score()`: Autoencoder reconstruction error (if model available)
- `load_autoencoder()`: Loads optional convolutional autoencoder
- `save_anomaly_frame()`: Saves detected anomaly frames to disk
- `save_audio_snippet()`: Saves anomaly audio snippets

**Class: `ConvAutoencoder`**
- Small convolutional autoencoder for reconstruction-based anomaly detection
- Encoder-decoder architecture with 3 layers each
- Trained on normal data, high reconstruction error indicates anomalies

**Scoring Methods**:
1. **Statistical**: Z-score based (simple, fast)
2. **Spectral**: Energy and frequency-based (audio-specific)
3. **Reconstruction**: Autoencoder MSE (learned patterns)

---

#### `utils/sync.py`
**Purpose**: Temporal alignment of audio and video

**Function: `align_audio_to_frames()`**
- Aligns audio windows to video frame timestamps
- Creates overlapping audio windows for each frame
- Handles edge cases (beginning/end of recording)
- Ensures consistent window sizes with padding

**Algorithm**:
1. For each frame timestamp, calculate audio window center
2. Extract audio segment around that timestamp
3. Pad if necessary to maintain consistent length
4. Return list of aligned audio windows

**Critical for**: Cross-modal fusion requiring synchronized features

---

#### `utils/config.py`
**Purpose**: Configuration management

**Function: `load_config()`**
- Loads YAML configuration file
- Handles 'auto' device selection (CUDA if available, else CPU)
- Returns configuration dictionary

---

#### `ui/app.py``
**Purpose**: Main Streamlit application and user interface

**Key Components**:

1. **Component Loading** (`load_components()`):
   - Caches model loading for performance
   - Initializes all models (detector, tracker, segmenter, audio, fusion)

2. **Visualization Functions**:
   - `overlay_boxes()`: Draws detection boxes with color coding
   - `plot_spectrogram()`: Visualizes mel spectrograms
   - `plot_scalogram()`: Visualizes CWT scalograms

3. **Main Processing Loop**:
   - Real-time frame capture and processing
   - Background thread for frame analysis
   - Audio streaming via callback
   - Live visualization updates

4. **Event Analysis**:
   - Combines visual and audio events
   - Industrial monitoring mode
   - General analysis mode
   - Detailed event logging

5. **Post-Processing**:
   - Detailed analysis after capture
   - Feature extraction and fusion
   - Trend visualization
   - Statistics dashboard

**Features**:
- Real-time video feed with annotations
- Audio classification display
- Event log with timestamps
- Anomaly trend charts
- Interactive mode selection
- Camera/microphone troubleshooting

---

## Dataflow: Input to Output

### Complete Processing Pipeline

#### Phase 1: Data Acquisition
```
Camera → capture_burst() → [Frame1, Frame2, ..., FrameN]
Microphone → record_audio() → Audio Array (numpy)
```

**Timestamps**: Both streams tagged with precise timestamps for synchronization

---

#### Phase 2: Preprocessing

**Visual Preprocessing**:
```
Raw Frame
    ↓
denoise_frame() → Denoised Frame
    ↓
cv2.cvtColor(COLOR_BGR2GRAY) → Grayscale
    ↓
compute_wavelet_energy() → Texture Energy Score
    ↓
optical_flow_tvl1(prev, curr) → Motion Flow Vectors
```

**Audio Preprocessing**:
```
Raw Audio Array
    ↓
_prepare_audio() → Resampled to 16kHz, Normalized
    ↓
compute_mel_spectrogram() → Mel Spectrogram (64 bins)
    ↓
compute_cwt_scalogram() → Wavelet Scalogram
    ↓
spectral_features() → {centroid, bandwidth, rolloff, zcr}
```

---

#### Phase 3: Visual Analysis

**Detection Pipeline**:
```
Preprocessed Frame
    ↓
FastDetector.detect() → Bounding Boxes
    ├─ YOLO Detection → Fast detections
    ├─ DETR Detection → Accurate detections (optional)
    └─ NMS → Deduplicated boxes
    ↓
ByteTrackWrapper.track_frames() → Tracked Objects
    └─ Persistent IDs across frames
    ↓
AdvancedSegmenter.segment_boxes() → Pixel Masks
    └─ SAM-based precise segmentation
```

**Motion Analysis**:
```
Frame Sequence
    ↓
optical_flow_tvl1() → Dense Flow Field
    ↓
cv2.cartToPolar() → Magnitude & Direction
    ↓
Motion Regions Detection → Moving Areas
```

---


#### Phase 4: Audio Analysis

**Multi-Model Audio Processing**:
```
Preprocessed Audio
    ↓
┌─────────────────────────────────────────┐
│  AdvancedAudioAnalyzer.analyze()        │
├─────────────────────────────────────────┤
│                                         │
│  AST Model                              │
│    ↓                                    │
│  Audio Event Classifications            │
│    ↓                                    │
│  AST Embeddings (768-dim)              │
│                                         │
│  Wav2Vec2 Model                         │
│    ↓                                    │
│  Speech Transcription                   │
│    ↓                                    │
│  Wav2Vec2 Embeddings (768-dim)         │
│                                         │
│  HuBERT Model                           │
│    ↓                                    │
│  Acoustic Scene Understanding           │
│    ↓                                    │
│  HuBERT Embeddings (768-dim)           │
│                                         │
│  Fusion Layer                           │
│    ↓                                    │
│  Combined Embeddings (256-dim)         │
└─────────────────────────────────────────┘
    ↓
{audio_events, transcription, embeddings, top_event}
```

---

#### Phase 5: Feature Tokenization

**Visual Tokens** (per frame):
```python
visual_token = [
    num_boxes,           # Number of detected objects
    mean_confidence,     # Average detection confidence
    wavelet_energy,      # Texture energy
    flow_magnitude,      # Motion magnitude
    flow_direction_x,    # Motion direction X
    flow_direction_y,     # Motion direction Y
    segmentation_area,  # Total segmented area
    track_count         # Number of active tracks
]
```

**Audio Tokens** (per frame, aligned):
```python
audio_token = [
    zcr,                # Zero-crossing rate
    centroid,           # Spectral centroid
    bandwidth,          # Spectral bandwidth
    rolloff,            # Spectral rolloff
    mel_energy,         # Mel spectrogram energy
    cwt_energy,         # CWT scalogram energy
    ast_confidence,     # AST model confidence
    audio_anomaly_score # Anomaly score
]
```

**Alignment**:
```
Frame Timestamps → align_audio_to_frames() → Audio Windows
    ↓
Each frame gets corresponding audio window
```

---

#### Phase 6: Cross-Modal Fusion

**Fusion Process**:
```
Visual Tokens [B, T, 8] + Audio Tokens [B, T, 8]
    ↓
Project to Hidden Dimension (256)
    ↓
Add Positional Embeddings
    ↓
┌─────────────────────────────────────────┐
│  Cross-Attention Layers (4 layers)      │
│                                         │
│  For each layer:                        │
│    Visual → Audio Attention             │
│    Audio → Visual Attention             │
│    Feed-Forward Network                 │
│    Layer Normalization                  │
└─────────────────────────────────────────┘
    ↓
Sequence Pooling (mean over time)
    ↓
Task Heads:
    ├─ Motion Classifier → [moving, static]
    └─ Event Classifier → [32 event categories]
```

---

#### Phase 7: Anomaly Detection

**Multi-Method Scoring**:
```
Frame + Audio
    ↓
┌─────────────────────────────────────────┐
│  Anomaly Scoring:                       │
│                                         │
│  1. Image Anomaly Score                 │
│     - Z-score based                    │
│     - Pixel intensity statistics        │
│                                         │
│  2. Autoencoder Score (if available)   │
│     - Reconstruction error              │
│     - Learned normal patterns           │
│                                         │
│  3. Audio Anomaly Score                 │
│     - Energy + spectral centroid       │
│                                         │
│  4. Event-Based Detection               │
│     - Detected anomaly classes         │
│     - Combined visual-audio events      │
└─────────────────────────────────────────┘
    ↓
Combined Anomaly Decision
```

---

#### Phase 8: Output & Visualization

**Real-Time Display**:
```
Processed Results
    ↓
┌─────────────────────────────────────────┐
│  Streamlit Dashboard:                   │
│                                         │
│  - Live Video Feed (annotated)          │
│  - Detection Boxes (color-coded)        │
│  - Motion Regions (overlay)             │
│  - Audio Classification                 │
│  - Event Analysis                       │
│  - Anomaly Alerts                       │
│  - Progress Bar                         │
└─────────────────────────────────────────┘
```

**Post-Processing Output**:
```
Complete Analysis
    ↓
┌─────────────────────────────────────────┐
│  Final Dashboard:                       │
│                                         │
│  - Event Log (DataFrame)                │
│  - Anomaly Trend Chart                  │
│  - Statistics Summary                   │
│  - Saved Anomaly Artifacts:             │
│    ├─ anomalies/frames/*.jpg            │
│    └─ anomalies/audio/*.wav             │
└─────────────────────────────────────────┘
```

---

## Libraries & Dependencies

### Core Deep Learning Frameworks

1. **PyTorch (torch, torchvision, torchaudio)**
   - **Version**: 2.4.1
   - **Purpose**: Neural network framework for all models
   - **Usage**: Model inference, tensor operations, GPU acceleration
   - **Key Modules**: `nn`, `torch.no_grad()`, device management

2. **Transformers (HuggingFace)**
   - **Version**: 4.45.2
   - **Purpose**: Pre-trained audio models (AST, Wav2Vec2, HuBERT)
   - **Usage**: Audio classification, feature extraction
   - **Key Models**: `AutoModelForAudioClassification`, `Wav2Vec2ForCTC`, `HubertForSequenceClassification`

3. **Ultralytics YOLO**
   - **Version**: 8.3.30
   - **Purpose**: Object detection and tracking
   - **Usage**: Fast object detection, ByteTrack tracking
   - **Key Features**: YOLOv8 models, built-in tracking

4. **Segment Anything Model (SAM)**
   - **Source**: Facebook Research (GitHub)
   - **Purpose**: Instance segmentation
   - **Usage**: Precise mask generation for detected objects
   - **Variants**: MobileSAM (lightweight), FastSAM (alternative)

### Computer Vision Libraries

5. **OpenCV (opencv-contrib-python)**
   - **Version**: 4.10.0.84
   - **Purpose**: Image processing, video I/O, optical flow
   - **Key Features**:
     - `cv2.fastNlMeansDenoisingColored()`: Image denoising
     - `cv2.calcOpticalFlowFarneback()`: Dense optical flow
     - `cv2.optflow.DualTVL1OpticalFlow_create()`: TV-L1 optical flow
     - Camera interface, frame capture

6. **scikit-image**
   - **Version**: 0.24.0
   - **Purpose**: Additional image processing utilities
   - **Usage**: Advanced image analysis operations

### Audio Processing Libraries

7. **Librosa**
   - **Version**: 0.10.2.post1
   - **Purpose**: Audio analysis and feature extraction
   - **Key Functions**:
     - `librosa.feature.melspectrogram()`: Mel spectrograms
     - `librosa.feature.spectral_centroid()`: Spectral features
     - `librosa.feature.zero_crossing_rate()`: Temporal features

8. **SoundDevice**
   - **Version**: 0.4.7
   - **Purpose**: Real-time audio I/O
   - **Usage**: Microphone recording, audio streaming
   - **Key Features**: Cross-platform audio interface, callback-based streaming

9. **SoundFile**
   - **Version**: 0.12.1
   - **Purpose**: Audio file I/O
   - **Usage**: Saving audio snippets (WAV files)

10. **PyWavelets**
    - **Version**: 1.6.0
    - **Purpose**: Wavelet transforms
    - **Usage**: 
      - Image: Texture energy via `pywt.wavedec2()`
      - Audio: Continuous Wavelet Transform via `pywt.cwt()`

### Signal Processing

11. **SciPy**
    - **Version**: 1.13.1
    - **Purpose**: Scientific computing and signal processing
    - **Key Modules**:
      - `scipy.signal.stft()`: Short-Time Fourier Transform
      - `scipy.signal.resample()`: Audio resampling
      - `scipy.stats.zscore()`: Statistical anomaly detection

12. **NumPy**
    - **Version**: >=1.26, <2.3
    - **Purpose**: Numerical computing, array operations
    - **Usage**: All data structures, mathematical operations

### Machine Learning Utilities

13. **scikit-learn**
    - **Version**: 1.5.2
    - **Purpose**: Machine learning utilities
    - **Usage**: Additional ML algorithms if needed

14. **timm**
    - **Version**: 0.9.12
    - **Purpose**: PyTorch image models
    - **Usage**: Additional vision model backbones

15. **einops**
    - **Version**: 0.8.0
    - **Purpose**: Tensor operations with readable syntax
    - **Usage**: `rearrange()` for tensor reshaping in transformers

16. **accelerate**
    - **Version**: 1.0.1
    - **Purpose**: HuggingFace acceleration utilities
    - **Usage**: Model optimization and distributed inference

### User Interface

17. **Streamlit**
    - **Version**: 1.39.0
    - **Purpose**: Web-based interactive dashboard
    - **Key Features**:
      - Real-time video display
      - Interactive controls
      - Data visualization
      - Caching for performance

### Visualization

18. **Matplotlib**
    - **Version**: 3.9.2
    - **Purpose**: Plotting and visualization
    - **Usage**: Spectrograms, scalograms, charts

19. **Pillow (PIL)**
    - **Version**: 10.4.0
    - **Purpose**: Image manipulation
    - **Usage**: Image processing utilities

### Utilities

20. **PyYAML**
    - **Version**: 6.0.2
    - **Purpose**: Configuration file parsing
    - **Usage**: Loading `config.yaml`

21. **supervision**
    - **Version**: 0.18.0
    - **Purpose**: Computer vision utilities
    - **Usage**: Additional CV helper functions

22. **ONNXRuntime**
    - **Version**: 1.19.2
    - **Purpose**: ONNX model inference (optional)
    - **Usage**: Alternative model deployment

23. **Pandas**
    - **Purpose**: Data manipulation and display
    - **Usage**: Event log DataFrame, statistics

---

## Innovative Features & Ideas

### 1. Multi-Model Audio Ensemble
**Innovation**: Combining three different audio models (AST, Wav2Vec2, HuBERT) for comprehensive audio understanding.

**Why Innovative**:
- Each model specializes in different aspects (classification, speech, scene)
- Fusion layer learns optimal combination of embeddings
- More robust than single-model approaches

**Implementation**: `models/audio_model.py` - `AdvancedAudioAnalyzer` class

---

### 2. Hybrid Object Detection
**Innovation**: Dual-detector system combining YOLO (speed) and DETR (accuracy) with cross-detector NMS.

**Why Innovative**:
- Balances real-time performance with detection accuracy
- NMS prevents duplicate detections across models
- Configurable to use both or just YOLO for speed

**Implementation**: `models/detector.py` - `AdvancedDetector` class

---

### 3. Bidirectional Cross-Modal Attention
**Innovation**: Transformer architecture where visual and audio modalities attend to each other bidirectionally.

**Why Innovative**:
- Visual features can inform audio understanding (e.g., seeing a crash helps identify crash sounds)
- Audio features can guide visual attention (e.g., sounds help locate visual events)
- More sophisticated than simple concatenation or late fusion

**Implementation**: `models/fusion.py` - `AdvancedCrossModalTransformer` class

---

### 4. Temporal Feature Alignment
**Innovation**: Precise synchronization of audio windows to video frames using timestamps.

**Why Innovative**:
- Ensures audio-visual features correspond to the same moment
- Handles variable frame rates and audio sampling
- Critical for accurate cross-modal fusion

**Implementation**: `utils/sync.py` - `align_audio_to_frames()` function

---

### 5. Multi-Resolution Analysis
**Innovation**: Combining multiple time-frequency representations (STFT, Mel, CWT) for comprehensive audio analysis.

**Why Innovative**:
- Different transforms capture different aspects:
  - STFT: Standard frequency analysis
  - Mel: Perceptual frequency scale
  - CWT: Multi-resolution time-frequency
- More complete feature representation

**Implementation**: `preprocess/audio.py` - Multiple transform functions

---

### 6. Wavelet-Based Texture Analysis
**Innovation**: Using wavelet decomposition for texture energy extraction in images.

**Why Innovative**:
- Captures multi-scale texture information
- More informative than simple pixel statistics
- Helps detect anomalies through texture changes

**Implementation**: `preprocess/image.py` - `compute_wavelet_energy()` function

---

### 7. Real-Time Background Processing
**Innovation**: Separate thread for frame analysis to maintain real-time video display.

**Why Innovative**:
- Prevents UI freezing during heavy computation
- Queue-based architecture for smooth processing
- Maintains responsive user experience

**Implementation**: `ui/app.py` - `frame_processor()` function and threading

---

### 8. Multi-Method Anomaly Detection
**Innovation**: Combining statistical, learned (autoencoder), and event-based anomaly detection.

**Why Innovative**:
- Different methods catch different types of anomalies
- Statistical: Unusual pixel/audio values
- Learned: Patterns not seen in training
- Event-based: Specific anomaly classes

**Implementation**: `utils/anomaly.py` - Multiple scoring functions

---

### 9. Industrial Monitoring Mode
**Innovation**: Specialized analysis mode focusing on industrial safety events.

**Why Innovative**:
- Tailored event categories (machine, tool, smoke, fire, leak, damage)
- Industrial sound classification
- Safety-focused alerting

**Implementation**: `ui/app.py` - Mode selection and event categorization

---

### 10. Adaptive Device Selection
**Innovation**: Automatic GPU/CPU selection with fallback and resource management.

**Why Innovative**:
- Works on systems with or without GPU
- Prevents OOM errors on low-resource systems
- Configurable thread limits for thermal management

**Implementation**: `utils/config.py` and `ui/app.py` - Device management

---

## Installation & Setup

### Prerequisites

- **Python**: 3.8 or higher (3.9+ recommended)
- **Operating System**: Windows, Linux, or macOS
- **Hardware**:
  - Webcam (USB or built-in)
  - Microphone (built-in or external)
  - GPU (optional, but recommended for faster processing)
  - RAM: Minimum 8GB (16GB+ recommended)
  - Storage: ~5GB for models and dependencies

### Step-by-Step Installation

#### 1. Clone or Navigate to Project Directory
```bash
cd CourseProject
```

#### 2. Create Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

#### 3. Upgrade pip
```bash
python -m pip install --upgrade pip
```

#### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: This will download and install all required packages. The first installation may take 10-20 minutes depending on your internet connection.

#### 5. Download Model Weights (Optional but Recommended)

**YOLO Model**:
- Automatically downloaded on first use
- Default: `yolov8n.pt` (nano version for speed)
- Can be changed in config to yolov8s, yolov8m, yolov8l, yolov8x

**MobileSAM Weights**:
- Download from: [MobileSAM GitHub](https://github.com/ChaoningZhang/MobileSAM)
- Place `mobile_sam.pt` in project root
- If not available, system uses dummy segmenter

**Autoencoder Weights (Optional)**:
- Train your own or use pre-trained weights
- Place at `models/ae.pth` for reconstruction-based anomaly detection

#### 6. Verify Installation
```bash
python -c "import torch; import cv2; import streamlit; print('Installation successful!')"
```

---

## Running the Project

### Basic Usage

#### 1. Start the Application
```bash
# Make sure virtual environment is activated
streamlit run ui/app.py
```

#### 2. Browser Opens Automatically
- Default URL: `http://localhost:8501`
- If browser doesn't open, navigate manually to the URL shown in terminal

#### 3. Camera and Microphone Setup
- The application will automatically detect available cameras
- If no camera is found, troubleshooting guide will be displayed
- Microphone defaults to device index 0 (can be changed in code)

#### 4. Start Analysis
- Click **"Start Analysis"** button
- The system will capture and process data for 10 seconds (configurable)
- Real-time results will be displayed

### Advanced Usage

#### Custom Configuration
Edit `configs/config.yaml` to customize:
- Detection confidence thresholds
- Frame capture settings
- Audio parameters
- Model selection
- Fusion network hyperparameters

#### Command-Line Options
```bash
# Run with custom port
streamlit run ui/app.py --server.port 8502

# Run with custom theme
streamlit run ui/app.py --theme.base light
```

### Running Modes

#### 1. Industrial Monitor Mode
- Focuses on industrial safety events
- Detects: machine, tool, smoke, fire, leak, damage
- Industrial sound classification
- Safety-focused alerts

#### 2. General Analysis Mode
- Comprehensive environment analysis
- All object types
- Detailed audio pattern analysis
- Complete event logging

#### Detail Levels

**Basic**:
- Core detections only
- Essential events
- Fast processing

**Advanced**:
- Detailed analysis
- Audio pattern matching
- Top sound predictions
- Comprehensive event correlation

### Processing Workflow

1. **Initialization** (5-10 seconds):
   - Load all models
   - Initialize camera and microphone
   - Set up processing pipeline

2. **Real-Time Processing** (10 seconds default):
   - Capture frames at ~15 FPS
   - Stream audio continuously
   - Process in background thread
   - Display results in real-time

3. **Post-Processing** (5-10 seconds):
   - Detailed analysis of all captured data
   - Feature extraction
   - Cross-modal fusion
   - Generate final reports

4. **Results Display**:
   - Event log table
   - Anomaly trend charts
   - Statistics summary
   - Saved anomaly artifacts

---

## Configuration

### Configuration File: `configs/config.yaml`

#### Device Settings
```yaml
device: auto  # Options: 'auto', 'cuda', 'cpu'
```
- `auto`: Automatically selects CUDA if available, else CPU
- `cuda`: Force GPU usage (requires CUDA-capable GPU)
- `cpu`: Force CPU usage (slower but works everywhere)

#### Burst Capture Settings
```yaml
burst:
  num_frames: 10          # Number of frames to capture
  frame_interval_ms: 50   # Milliseconds between frames (~20 FPS)
```

#### Audio Settings
```yaml
audio:
  sample_rate: 16000      # Audio sampling rate (Hz)
  duration_sec: 2.0       # Audio recording duration
  channels: 1             # Mono (1) or Stereo (2)
```

#### Vision Settings
```yaml
vision:
  detector: yolov8n      # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  conf_threshold: 0.3     # Detection confidence threshold (0.0-1.0)
  iou_threshold: 0.5      # Intersection over Union for NMS
  classes: null           # Specific classes to detect (null = all)
  tracker: bytetrack      # Tracking algorithm
  segmentation_model: mobile_sam  # Options: mobile_sam, fast_sam
  flow: tvl1              # Optical flow method
```

#### Audio Model Settings
```yaml
audio_model:
  hub_model: MIT/ast-finetuned-audioset-10-10-0.4593  # HuggingFace model ID
```

#### Fusion Network Settings
```yaml
fusion:
  hidden_dim: 128         # Hidden dimension size
  num_layers: 2           # Number of transformer layers
  num_heads: 4            # Number of attention heads
  dropout: 0.1            # Dropout rate
```

#### UI Settings
```yaml
ui:
  max_width: 1200         # Maximum UI width (pixels)
  theme: light            # Theme: light or dark
```

### Environment Variables (Optional)

You can set these before running:
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""

# Set number of threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Camera Not Detected

**Symptoms**: Error message about no cameras found

**Solutions**:
- Check camera is connected and powered
- Close other applications using the camera (Zoom, Teams, etc.)
- Check Windows privacy settings (Settings > Privacy > Camera)
- Try different USB ports
- Update camera drivers
- Restart computer

**Windows-Specific**:
- Check Device Manager for camera issues
- Verify camera works in Windows Camera app
- Grant camera permissions to Python/Streamlit

#### 2. Microphone Not Working

**Symptoms**: No audio input, audio errors

**Solutions**:
- Check microphone is connected and selected
- Verify microphone permissions
- Test microphone in other applications
- Check system audio settings
- Update audio drivers

#### 3. Out of Memory Errors

**Symptoms**: CUDA out of memory, system slowdown

**Solutions**:
- Reduce `num_frames` in config
- Use smaller YOLO model (yolov8n instead of yolov8x)
- Set `device: cpu` in config
- Close other applications
- Reduce frame resolution in code (FRAME_W, FRAME_H)

#### 4. Slow Performance

**Symptoms**: Low FPS, laggy interface

**Solutions**:
- Use GPU if available (`device: cuda`)
- Reduce frame resolution
- Use smaller models (yolov8n, mobile_sam)
- Disable DETR detector (set `enable_detr: False`)
- Reduce number of frames processed

#### 5. Model Download Errors

**Symptoms**: Failed to load models, network errors

**Solutions**:
- Check internet connection
- Manually download models:
  - YOLO: Models auto-download from Ultralytics
  - HuggingFace: Models auto-download from HuggingFace Hub
- Use VPN if behind firewall
- Increase timeout in code if needed

#### 6. Import Errors

**Symptoms**: ModuleNotFoundError, ImportError

**Solutions**:
- Verify virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- Check Python version (3.8+)
- Verify all dependencies installed correctly

#### 7. Streamlit Issues

**Symptoms**: App won't start, port already in use

**Solutions**:
- Kill existing Streamlit processes
- Use different port: `streamlit run ui/app.py --server.port 8502`
- Clear Streamlit cache: `streamlit cache clear`
- Restart terminal/IDE

#### 8. Audio Processing Errors

**Symptoms**: Audio analysis fails, warnings about audio

**Solutions**:
- Check audio sample rate matches config (16000 Hz)
- Verify audio is not all zeros (silence)
- Ensure audio length is sufficient
- Check audio normalization

#### 9. Segmentation Model Not Found

**Symptoms**: Warning about missing SAM weights, using dummy segmenter

**Solutions**:
- Download MobileSAM weights: `mobile_sam.pt`
- Place in project root directory
- Or use FastSAM: Download `sam_vit_h_4b8939.pth`
- System will work with dummy segmenter (less accurate)

#### 10. CUDA/GPU Issues

**Symptoms**: CUDA errors, GPU not detected

**Solutions**:
- Verify CUDA is installed: `nvidia-smi`
- Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-enabled PyTorch if needed
- Use CPU mode: `device: cpu` in config

### Performance Optimization Tips

1. **For Speed**:
   - Use `yolov8n` (nano) model
   - Disable DETR detector
   - Use `mobile_sam` for segmentation
   - Reduce frame resolution
   - Process fewer frames

2. **For Accuracy**:
   - Use `yolov8x` (extra large) model
   - Enable DETR detector
   - Use full SAM model
   - Increase frame resolution
   - Process more frames

3. **For Memory Efficiency**:
   - Use CPU mode
   - Reduce batch sizes
   - Process frames sequentially
   - Limit stored frames
   - Clear cache regularly

### Getting Help

If issues persist:
1. Check error messages carefully
2. Review logs in terminal
3. Verify all dependencies are correct versions
4. Test individual components separately
5. Check GitHub issues (if project is on GitHub)
6. Review model documentation (YOLO, SAM, Transformers)

---

## Additional Notes

### Model Weights Location
- YOLO weights: Auto-downloaded to `~/.ultralytics/`
- HuggingFace models: Auto-downloaded to `~/.cache/huggingface/`
- SAM weights: Project root (if manually downloaded)
- Autoencoder: `models/ae.pth` (if available)

### Output Files
- Anomaly frames: `anomalies/frames/*.jpg`
- Anomaly audio: `anomalies/audio/*.wav`
- Logs: Check Streamlit terminal output

### Extending the System
- Add new detection classes: Modify `DETECTION_CLASSES` in `ui/app.py`
- Add new audio events: Modify `AUDIO_EVENTS` in `ui/app.py`
- Custom anomaly detection: Extend `utils/anomaly.py`
- New fusion methods: Modify `models/fusion.py`
- Additional sensors: Extend `sensors/` directory

---

## License

This project assembles open-source components. Please respect the licenses of:
- YOLO (Ultralytics): AGPL-3.0
- SAM (Meta): Apache 2.0
- Transformers (HuggingFace): Apache 2.0
- Other dependencies: Check individual licenses

---

## Acknowledgments

- **Ultralytics** for YOLO models
- **Meta AI** for Segment Anything Model
- **HuggingFace** for Transformers library and audio models
- **OpenCV** community for computer vision tools
- **Streamlit** for the web framework

---

## Version Information

- **Project Version**: 1.0
- **Last Updated**: 2024
- **Python**: 3.8+
- **PyTorch**: 2.4.1
- **Streamlit**: 1.39.0

---

**For detailed technical questions or contributions, please refer to the code documentation and inline comments.**
