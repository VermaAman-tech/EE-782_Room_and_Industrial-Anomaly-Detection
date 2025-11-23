import io
import os
import sys
import time
import warnings
from typing import List
from queue import Queue
from threading import Thread
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import gc
import sounddevice as sd
import collections
import pandas as pd

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Overwriting.*in registry')
warnings.filterwarnings('ignore', message='Some weights of.*were not used')
warnings.filterwarnings('ignore', message='Some weights of.*were not initialized')
warnings.filterwarnings('ignore', message=".*par-call bakcend dispatch.*")

# Ensure project root is on sys.path when launched via Streamlit from ui/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.detector import FastDetector
from models.tracker import ByteTrackWrapper
from models.segmenter import get_segmenter
from models.audio_model import AudioClassifierAST
from models.fusion import build_fusion_head
from preprocess.image import denoise_frame, compute_wavelet_energy, optical_flow_tvl1, draw_flow_arrows
from preprocess.audio import compute_mel_spectrogram, compute_cwt_scalogram, spectral_features
from sensors.camera import list_cameras, capture_burst
from sensors.mic import list_microphones, record_audio
from utils.config import load_config
from utils.sync import align_audio_to_frames
from utils.anomaly import (
    image_anomaly_score,
    audio_anomaly_score,
    load_autoencoder,
    ae_anomaly_score,
    save_anomaly_frame,
    save_audio_snippet,
)
import pandas as pd


@st.cache_resource
def load_components(cfg):
    device = cfg['device']
    detector = FastDetector(cfg['vision']['detector'], device=device, conf=cfg['vision']['conf_threshold'], iou=cfg['vision']['iou_threshold'], classes=cfg['vision']['classes'])
    tracker = ByteTrackWrapper(cfg['vision']['detector'], device=device, conf=cfg['vision']['conf_threshold'], iou=cfg['vision']['iou_threshold'], classes=cfg['vision']['classes'])
    segmenter = get_segmenter(cfg['vision']['segmentation_model'])
    audio_model = AudioClassifierAST(cfg['audio_model']['hub_model'], device=device)
    fusion_head = build_fusion_head(visual_dim=8, audio_dim=8, cfg=cfg).to(device)  # small dims for demo features
    fusion_head.eval()
    return detector, tracker, segmenter, audio_model, fusion_head


def overlay_boxes(frame: np.ndarray, boxes: List[dict], motion_regions=None) -> np.ndarray:
    """Overlay detection boxes and motion regions with enhanced visualization"""
    vis = frame.copy()
    
    # Draw motion regions first (semi-transparent)
    if motion_regions:
        motion_overlay = np.zeros_like(frame, dtype=np.uint8)
        for region in motion_regions:
            x1, y1, x2, y2 = map(int, region['box'])
            confidence = region['confidence']
            # Brighter color for higher confidence
            color = (0, int(min(255, confidence * 2.55)), 0)
            cv2.rectangle(motion_overlay, (x1, y1), (x2, y2), color, -1)
        
        # Blend motion overlay
        alpha = 0.3
        vis = cv2.addWeighted(vis, 1.0, motion_overlay, alpha, 0)
    
    # Draw detected objects
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.get('box', b.get('xyxy', [0,0,0,0])))
        conf = b.get('conf', 0.0)
        obj_type = b.get('type', 'unknown')
        
        # Color coding: red for anomalies, blue for normal objects
        is_anomaly = obj_type in ['smoke', 'fire', 'leak', 'damage', 'break']
        is_moving = b.get('motion', False)
        
        if is_anomaly:
            color = (0, 0, 255)  # Red for anomalies
        elif is_moving:
            color = (255, 165, 0)  # Orange for moving objects
        else:
            color = (255, 0, 0)  # Blue for normal objects
        
        # Draw box with thickness based on confidence
        thickness = max(1, int(conf * 3))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        
        # Enhanced label with confidence
        label = f"{obj_type} ({conf:.2f})"
        if is_moving:
            label += " [MOVING]"
        
        # Background for text
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1-label_h-5), (x1+label_w, y1), color, -1)
        cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis


def plot_spectrogram(S_db: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(2, 1.5))
    im = ax.imshow(S_db, origin='lower', aspect='auto', cmap='magma')
    ax.set_title('Mel-Spectrogram', fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    return buf.getvalue()


def plot_scalogram(cwt: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(2, 1.5))
    im = ax.imshow(np.abs(cwt), origin='lower', aspect='auto', cmap='viridis')
    ax.set_title('CWT Scalogram', fontsize=8)
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    return buf.getvalue()


def main():
    # Set the page configuration at the very start
    if 'page_config_done' not in st.session_state:
        st.set_page_config(page_title="Multimodal Room Monitor", layout="wide", initial_sidebar_state="collapsed")
        st.session_state.page_config_done = True
        
    def overlaps(box1, box2, thresh=0.3):
        """Check if two bounding boxes overlap significantly"""
        if len(box1) == 4 and len(box2) == 4:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            if x2 <= x1 or y2 <= y1:
                return False
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            return intersection / min(area1, area2) > thresh
        return False

    # Custom CSS to reduce spacing and font sizes
    st.markdown("""
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        .element-container {margin: 0.5rem 0px;}
        h1 {font-size: 1.5rem !important; margin: 0.2rem 0 !important;}
        h2 {font-size: 1.2rem !important; margin: 0.2rem 0 !important;}
        h3 {font-size: 1rem !important; margin: 0.2rem 0 !important;}
        .stButton>button {padding: 0.2rem 1rem;}
        </style>
    """, unsafe_allow_html=True)
    
    cfg = load_config('configs/config.yaml')
    # Force device selection to avoid GPU OOM on small laptops unless explicitly allowed
    if torch.cuda.is_available() and cfg.get('allow_cuda', False):
        cfg['device'] = 'cuda'
    else:
        cfg['device'] = 'cpu'
    # Limit CPU threads to reduce thermal/ram pressure on laptops
    try:
        torch.set_num_threads(cfg.get('torch_threads', 1))
    except Exception:
        pass
    st.title('Multimodal Room Monitor')

    # Camera and mic setup with enhanced error handling
    try:
        # First check Windows privacy settings
        if os.name == 'nt':  # Windows
            import winreg
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                   r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam", 
                                   0, winreg.KEY_READ)
                value, _ = winreg.QueryValueEx(key, "Value")
                if value == "Deny":
                    st.error("Camera access is disabled in Windows privacy settings.")
                    st.info("To enable camera access:\n"
                           "1. Open Windows Settings\n"
                           "2. Go to Privacy & Security > Camera\n"
                           "3. Enable 'Camera access' and 'Let apps access your camera'")
                    return
            except Exception:
                pass  # Key might not exist on all Windows versions
        
        # Try to detect cameras
        available_cameras = list_cameras()
        
        if not available_cameras:
            st.error("No working cameras detected. Attempting alternative detection methods...")
            
            # Try DirectShow enumeration on Windows
            if os.name == 'nt':
                import subprocess
                try:
                    # PowerShell command to list video devices
                    cmd = "Get-WmiObject Win32_PnPEntity | Where-Object {$_.Name -like '*Camera*' -or $_.Name -like '*Webcam*'} | Select-Object Name"
                    result = subprocess.run(['powershell', '-Command', cmd], capture_output=True, text=True)
                    if 'Camera' in result.stdout or 'Webcam' in result.stdout:
                        st.warning("Cameras found in system devices but can't be accessed. This usually indicates a permissions or driver issue.")
                except Exception:
                    pass
            
            st.info("""
            ### Camera Troubleshooting Guide
            
            1. **Check Physical Connection**
               - Ensure camera is properly plugged in
               - Try a different USB port
               - Look for LED indicators on the camera
            
            2. **Check System Settings**
               - Open Camera app to verify Windows can access it
               - Check Windows privacy settings
               - Verify no other apps are using the camera
            
            3. **Driver Issues**
               - Open Device Manager
               - Look for warning icons under 'Imaging devices' or 'Cameras'
               - Try updating or reinstalling camera drivers
            
            4. **Application Conflict Resolution**
               - Close other apps that might use the camera (Zoom, Teams, etc.)
               - Restart your browser
               - Clear browser camera permissions
            
            5. **If Problems Persist**
               - Restart your computer
               - Check for Windows updates
               - Update camera firmware if available
            """)
            return
            
        # Successfully found cameras
        cam_idx = available_cameras[0]
        st.success(f"Found {len(available_cameras)} camera(s). Using camera index {cam_idx}")
        
        # Test the selected camera thoroughly
        test_cap = cv2.VideoCapture(int(cam_idx))
        if not test_cap.isOpened():
            st.error(f"Found camera {cam_idx} but failed to open it. Trying DirectShow...")
            # Try DirectShow as fallback
            test_cap = cv2.VideoCapture(int(cam_idx) + cv2.CAP_DSHOW)
            if not test_cap.isOpened():
                st.error("Failed to open camera with both default and DirectShow backends")
                return
                
        # Verify we can read frames
        ret, frame = test_cap.read()
        if not ret or frame is None:
            st.error("Camera opened but cannot read frames. This might indicate a driver issue.")
            test_cap.release()
            return
            
        test_cap.release()
        mic_idx = 0  # TODO: Add mic detection similar to cameras
        
    except Exception as e:
        st.error(f"Unexpected error during camera setup: {str(e)}")
        st.info("Please try restarting the application or your computer.")
        return
    
    # Test camera access before proceeding
    test_cap = cv2.VideoCapture(int(cam_idx))
    if not test_cap.isOpened():
        st.error(f"Failed to open camera {cam_idx}. Please try another camera or restart the application.")
        return
    test_cap.release()
    
    detector, tracker, segmenter, audio_model, fusion_head = load_components(cfg)

    # Fixed capture settings
    live_seconds = 10  # Fixed 10 seconds
    burst_frames = cfg['burst']['num_frames']
    


    # Analysis control and display
    st.subheader("Real-time Event Analysis Dashboard")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        start_button = st.button("Start Analysis", type="primary")
        st.info(f"Camera {cam_idx} and microphone ready")
    
    with col2:
        mode_type = st.radio(
            "Analysis Mode",
            ["Industrial Monitor", "General Analysis"],
            help="Industrial: Focus on industrial anomalies and safety. General: Detailed environment analysis"
        )
        detail_level = st.radio(
            "Detail Level",
            ["Basic", "Advanced"],
            help="Basic: Core detections. Advanced: Detailed analysis with audio patterns"
        )
    
    status = st.empty()
    live_alert = st.empty()
    event_display = st.empty()  # For real-time event analysis
    progress_bar = st.progress(0)
    
    # Only start monitoring when button is clicked
    if not start_button:
        st.info("Click 'Start Analysis' to begin monitoring")
        st.stop()

    # Comprehensive event detection classes
    DETECTION_CLASSES = {
        # Objects
        0: 'person',
        1: 'bottle',
        2: 'container',
        3: 'tool',
        4: 'machine',
        5: 'vehicle',
        # Events
        6: 'fall',
        7: 'break',
        8: 'spill',
        9: 'collision',
        10: 'movement',
        # Conditions
        11: 'smoke',
        12: 'fire',
        13: 'leak',
        14: 'damage',
        15: 'normal'
    }
    
    # Detailed sound categories and causes
    AUDIO_EVENTS = {
        'impact': {
            'sounds': ['crash', 'bang', 'thud', 'smack', 'hit'],
            'causes': ['Object falling', 'Collision', 'Door slam', 'Tool drop', 'Surface impact']
        },
        'continuous': {
            'sounds': ['hum', 'whir', 'buzz', 'drone', 'hiss'],
            'causes': ['Machine operation', 'Fan/ventilation', 'Motor running', 'Equipment active', 'Air flow']
        },
        'alarm': {
            'sounds': ['beep', 'siren', 'alert', 'alarm', 'chirp'],
            'causes': ['Warning signal', 'Emergency alert', 'System notification', 'Safety alarm', 'Equipment alert']
        },
        'human': {
            'sounds': ['speech', 'cough', 'footsteps', 'clap', 'knock'],
            'causes': ['Person present', 'Movement nearby', 'Manual operation', 'Intentional signal', 'Human activity']
        },
        'mechanical': {
            'sounds': ['machine', 'engine', 'motor', 'pump', 'gear'],
            'causes': ['Equipment running', 'Machinery active', 'System operation', 'Moving parts', 'Mechanical process']
        },
        'liquid': {
            'sounds': ['drip', 'splash', 'flow', 'pour', 'spray'],
            'causes': ['Fluid leak', 'Liquid transfer', 'Spillage', 'Water flow', 'Fluid release']
        },
        'friction': {
            'sounds': ['screech', 'scratch', 'squeak', 'grind', 'rub'],
            'causes': ['Metal contact', 'Surface wear', 'Part misalignment', 'Material stress', 'Component friction']
        },
        'anomaly': {
            'sounds': ['crack', 'pop', 'snap', 'clunk', 'rattle'],
            'causes': ['Component failure', 'Material break', 'Structural issue', 'Part looseness', 'System fault']
        }
    }

    # Optimize capture settings for lower RAM/CPU usage (smaller frames, lower FPS)
    FRAME_W, FRAME_H = 320, 240  # Smaller frames to save memory and screen space
    FPS = int(min(15, cfg.get('max_fps', 15)))  # moderate FPS
    cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(int(cam_idx))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize camera buffer
    except Exception:
        pass

    # Audio settings for real-time processing
    sr = int(cfg['audio']['sample_rate'])
    channels = int(cfg['audio']['channels'])
    # Keep audio buffer short to avoid large memory usage (250ms - 500ms)
    audio_buf = collections.deque(maxlen=max(1, sr // 4))  # 0.25s buffer
    start_time = time.time()
    frame_interval = 1.0 / FPS

    def audio_cb(indata, frames, time_info, status):
        if status:
            pass
        audio_buf.append(indata.copy().squeeze(-1) if channels == 1 else indata.copy())

    stream = sd.InputStream(device=int(mic_idx), samplerate=sr, channels=channels, callback=audio_cb)
    stream.start()

    max_frames = int(live_seconds * FPS)
    # Cap stored frames to avoid huge memory usage; store only recent frames
    max_stored_frames = min(max_frames, 60)
    collected_frames = []
    frame_ts = []
    event_log = []
    anomaly_counts = {k: 0 for k in DETECTION_CLASSES.values()}
    # Load optional autoencoder (place weights at models/ae.pth)
    ae_weights = os.path.join(PROJECT_ROOT, 'models', 'ae.pth')
    device = cfg.get('device', 'cpu')
    try:
        ae = load_autoencoder(ae_weights, device=device)
    except Exception:
        ae = None
    # UI placeholders for live display
    live_video = st.empty()
    live_spec = st.empty()
    audio_label_box = st.empty()
    saved_count = 0
    # Very small queue to avoid accumulating frames in memory
    frame_queue = Queue(maxsize=1)
    result_queue = Queue()
    
    def analyze_event(visual_events, audio_events, frame):
        """Combine audio-visual events for detailed analysis"""
        combined_analysis = []
        
        # Match objects with sounds
        for v_event in visual_events:
            if v_event in ['bottle', 'container'] and any(a in audio_events for a in ['crash', 'bang', 'impact']):
                combined_analysis.append(f"{v_event} fall detected")
            elif v_event in ['machine', 'tool'] and any(a in audio_events for a in ['whir', 'buzz', 'mechanical']):
                combined_analysis.append(f"Active {v_event} operation")
            elif v_event == 'spill' and any(a in audio_events for a in ['splash', 'drip', 'liquid']):
                combined_analysis.append("Liquid spill detected")
            elif v_event in ['break', 'collision'] and any(a in audio_events for a in ['crash', 'bang', 'impact']):
                combined_analysis.append(f"Impact event: {v_event}")
        
        return combined_analysis

    def process_frame(frame, detector, ae, audio_model, audio_data=None, sr=None):
        """Process frame and audio in separate thread"""
        # Visual detection with YOLO
        dets = detector.detect([frame])[0]
        detected_types = []
        detected_objects = []
        motion_info = {'regions': [], 'description': 'No motion', 'confidence': 0.0}
        
        # Track previous frame for motion
        if len(collected_frames) > 1:
            prev = collected_frames[-1]
            # Motion detection using frame differencing
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                              pyr_scale=0.5, levels=5, winsize=15, 
                                              iterations=3, poly_n=5, poly_sigma=1.2, 
                                              flags=0)
            
            # Calculate motion magnitude and direction
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = magnitude > 1.0  # Threshold for significant motion
            
            if np.any(motion_mask):
                # Find motion regions
                motion_regions = []
                contours, _ = cv2.findContours((motion_mask * 255).astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    if cv2.contourArea(cnt) > 100:  # Filter small motions
                        x, y, w, h = cv2.boundingRect(cnt)
                        conf = float(np.mean(magnitude[y:y+h, x:x+w]))
                        motion_regions.append({
                            'box': [x, y, x+w, y+h],
                            'confidence': min(100.0, conf * 20.0)
                        })
                
                if motion_regions:
                    motion_info = {
                        'regions': motion_regions,
                        'description': 'Active motion detected',
                        'confidence': float(np.mean(magnitude[motion_mask]) * 20.0)
                    }
        
        # Object detection with confidence (prefer detector-provided human labels)
        for d in dets:
            cls_id = d.get('cls', -1)
            conf = float(d.get('conf', 0.0))
            if conf > 0.4:
                det_type = d.get('label') if d.get('label') is not None else DETECTION_CLASSES.get(cls_id, f'class_{cls_id}')
                detected_types.append(det_type)
                detected_objects.append({
                    'type': det_type,
                    'conf': conf,
                    'box': d.get('xyxy', [0, 0, 0, 0]),
                    'motion': any(overlaps(d.get('xyxy', [0,0,0,0]), r['box']) for r in motion_info['regions'])
                })
        
        # Motion detection using frame differencing
        if len(collected_frames) > 1:
            prev_frame = cv2.cvtColor(collected_frames[-1], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_frame, curr_frame)
            motion_score = np.mean(diff) / 255.0
            if motion_score > 0.05:  # Motion threshold
                detected_types.append('movement')
        
        # Audio event detection
        audio_events = []
        if audio_data is not None and sr is not None:
            try:
                # Get audio event classification
                a_label, a_prob, a_topk = audio_model.classify(audio_data, sr)
                if a_prob > 0.3:  # Confidence threshold
                    audio_events = [a_label]
                    # Add top-k predictions for more context
                    audio_events.extend([k for k, _ in a_topk[:3] if k != a_label])
            except Exception as e:
                pass  # Handle audio processing errors gracefully
        
        # Anomaly detection
        try:
            ae_score = ae_anomaly_score(ae, frame, device=device) if ae is not None else 0.0
        except:
            ae_score = float(image_anomaly_score(frame))
        
        # Combine audio-visual analysis
        event_analysis = analyze_event(detected_types, audio_events, frame)
            
        return {
            'dets': detected_objects,
            'types': detected_types,
            'ae_score': ae_score,
            'audio_events': audio_events,
            'analysis': event_analysis
        }
    
    def frame_processor():
        while True:
            frame = frame_queue.get()
            if frame is None:  # Sentinel to stop thread
                break
                
            # Get latest audio snippet
            current_audio = None
            current_sr = None
            if len(audio_buf) > 0:
                current_audio = np.concatenate(list(audio_buf))[-int(sr/4):]  # Last 250ms
                current_sr = sr
                
            results = process_frame(frame, detector, ae, audio_model, 
                                 audio_data=current_audio, sr=current_sr)
            result_queue.put(results)
            frame_queue.task_done()
    
    # Start processing thread
    processor_thread = Thread(target=frame_processor, daemon=True)
    processor_thread.start()
    
    # keep last results from worker to use for logging/saving when worker is slightly behind
    last_results = {'types': [], 'ae_score': 0.0, 'audio_events': [], 'dets': [], 'analysis': []}
    try:
        while len(collected_frames) < max_frames and (time.time() - start_time) < live_seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            ts = time.time()
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            
            # Process results from the analysis thread
            try:
                if not result_queue.empty():
                    results = result_queue.get_nowait()
                    # update last_results for downstream usage
                    last_results = results
                    
                    # Annotate frame with detections; mark MOVING and SOUND
                    vis_frame = frame.copy()
                    sound_flag = bool(results.get('audio_events'))
                    for obj in results['dets']:
                        x1, y1, x2, y2 = map(int, obj['box'])
                        conf = obj['conf']
                        moving = bool(obj.get('motion', False))
                        label = f"{obj['type']} ({conf:.2f})"
                        if moving:
                            label += " [MOVING]"
                        if sound_flag:
                            label += " [SOUND]"
                        color = (255, 165, 0) if moving else (0, 255, 0)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Display frame with annotations (smaller width to save screen space)
                    live_video.image(cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB),
                                     caption='Live Analysis', width=420)
                    # Update anomaly counts only (heatmap removed)
                    try:
                        for obj in results['dets']:
                            t = obj.get('type')
                            if t in anomaly_counts:
                                anomaly_counts[t] += 1
                    except Exception:
                        pass
                    
                    # Show real-time event analysis based on mode
                    event_text = []
                    
                    if mode_type == "Industrial Monitor":
                        # Focus on industrial events and anomalies
                        if results['types']:
                            industrial_events = [t for t in results['types'] if t in ['machine', 'tool', 'smoke', 'fire', 'leak', 'damage']]
                            if industrial_events:
                                event_text.append(f"⚠️ Industrial Events: {', '.join(industrial_events)}")
                        
                        if results['audio_events']:
                            industrial_sounds = []
                            for event in results['audio_events']:
                                for category in ['mechanical', 'alarm', 'anomaly', 'friction']:
                                    if event in AUDIO_EVENTS[category]['sounds']:
                                        cause = AUDIO_EVENTS[category]['causes'][0]  # Primary cause
                                        industrial_sounds.append(f"{event} ({cause})")
                            if industrial_sounds:
                                event_text.append(f"🔊 Industrial Sounds: {', '.join(industrial_sounds)}")
                    else:  # General Analysis
                        # Comprehensive environment analysis
                        if results['types']:
                            event_text.append(f"�️ Detected Objects: {', '.join(results['types'])}")
                        
                        if results['audio_events']:
                            audio_analysis = []
                            for event in results['audio_events']:
                                for category, details in AUDIO_EVENTS.items():
                                    if event in details['sounds']:
                                        causes = details['causes'][:2]  # Top 2 possible causes
                                        audio_analysis.append(f"{event}: {' or '.join(causes)}")
                            if audio_analysis:
                                event_text.append(f"🔊 Sound Analysis:\n" + "\n".join(f"  • {a}" for a in audio_analysis[:5]))
                    
                    # Add motion and anomaly information for both modes
                    if results.get('motion_info'):
                        motion = results['motion_info']
                        event_text.append(f"📱 Motion: {motion['description']} ({motion['confidence']:.1f}%)")
                    
                    if detail_level == "Advanced":
                        if results['analysis']:
                            event_text.append(f"🔍 Detailed Analysis:\n" + "\n".join(f"  • {a}" for a in results['analysis']))
                        if results.get('audio_confidence'):
                            top_sounds = results['audio_confidence'][:5]  # Top 5 possible sounds
                            event_text.append("🎯 Top Sound Matches:\n" + 
                                           "\n".join(f"  • {sound}: {conf:.1f}%" for sound, conf in top_sounds))
                    
                    if event_text:
                        event_display.info("\n".join(event_text))
                    else:
                        event_display.info("No significant events detected")
                    
                    # Alert on significant events
                    if results['analysis'] or results['ae_score'] > 0.02:
                        live_alert.warning("⚠️ Event Detected!")
                    else:
                        live_alert.info("✅ Normal Operation")
                    
                else:
                    # Show raw frame while waiting for analysis (compact)
                    live_video.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                     caption='Live Feed', width=360)
            except Exception as e:
                pass
                
            # Queue frame for processing
            if not frame_queue.full():
                frame_queue.put(frame.copy())
            frame_ts.append(ts)
            collected_frames.append(frame)
            # Cap stored frames to avoid high memory usage
            while len(collected_frames) > max_stored_frames:
                try:
                    collected_frames.pop(0)
                except Exception:
                    break

            # use last_results for logging and alerts
            detected_types = last_results.get('types', [])
            ae_score = float(last_results.get('ae_score', 0.0))
            audio_events_now = last_results.get('audio_events', [])
            # Cap stored frames to avoid high memory usage
            while len(collected_frames) > max_stored_frames:
                # drop oldest frame reference
                try:
                    collected_frames.pop(0)
                except Exception:
                    break

            # NOTE: visual detection is handled in the background worker (no duplicate detection here)

            # Audio anomaly detection
            if len(audio_buf) > 0:
                audio_now = np.concatenate(list(audio_buf), axis=0)
                aud_score = audio_anomaly_score(audio_now, sr)
                aud_anomaly = aud_score > 0.5
            else:
                aud_score = 0.0
                aud_anomaly = False

            # Audio classification (periodic, low-cost)
            if len(audio_buf) >= int(0.5 * sr):
                try:
                    last_audio = np.concatenate(list(audio_buf))[-int(0.5 * sr):]
                    a_label, a_prob, a_topk = audio_model.classify(last_audio, sr)
                    audio_label_box.info(f"Audio: {a_label} ({a_prob*100:.1f}%)")
                except Exception:
                    a_label, a_prob, a_topk = (None, 0.0, [])
            else:
                a_label, a_prob, a_topk = (None, 0.0, [])

            # Autoencoder anomaly score
            try:
                ae_score = ae_anomaly_score(ae, frame, device=device)
            except Exception:
                ae_score = float(image_anomaly_score(frame))
            img_anomaly = ae_score > 0.02 or len(detected_types) > 0

            # Log event
            event = {
                'timestamp': ts,
                'image_anomalies': detected_types,
                'ae_score': float(ae_score),
                'audio_anomaly': aud_anomaly,
                'audio_score': float(aud_score),
                'audio_label': a_label if a_label is not None else None,
            }
            event_log.append(event)

            # Live alert and save artifacts
            if detected_types or aud_anomaly or (ae_score > 0.02):
                live_alert.error(f"Anomaly Detected! Types: {', '.join(detected_types) if detected_types else 'None'} | Audio: {aud_anomaly} (score={aud_score:.2f}) | AE:{ae_score:.3f}")
                # Save frame and audio snippet
                try:
                    save_anomaly_frame(frame, ts, detected_types)
                    if len(audio_buf) > 0:
                        snippet = np.concatenate(list(audio_buf))[-sr:]
                        save_audio_snippet(snippet, sr, ts, label=','.join(detected_types) or 'audio')
                    saved_count += 1
                except Exception:
                    pass
            else:
                live_alert.info(f"Normal. Audio score={aud_score:.2f} | AE:{ae_score:.3f}")

            # Progress
            progress = min(1.0, (time.time() - start_time) / live_seconds)
            progress_bar.progress(progress)

            # Maintain frame rate
            frame_delay = frame_interval - (time.time() - ts)
            if frame_delay > 0:
                time.sleep(frame_delay)
    finally:
        # Clean up threads and resources
        frame_queue.put(None)  # Signal processing thread to stop
        processor_thread.join()
        cap.release()
        stream.stop(); stream.close()
        
    status.success('Capture complete! Running final analysis...')
    progress_bar.empty()

    # Convert audio buffer to array
    audio_all = np.concatenate(list(audio_buf), axis=0) if len(audio_buf) > 0 else np.zeros(int(sr * max(0.1, live_seconds)))

    # Post-capture: run detailed analysis using preexisting pipeline
    frames = collected_frames
    audio = audio_all

    denoised = [denoise_frame(f) for f in frames]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in denoised]
    wavelet_energy = [compute_wavelet_energy(g.astype(np.float32) / 255.0) for g in grays]

    flows = []
    for i in range(max(0, len(grays) - 1)):
        flows.append(optical_flow_tvl1(grays[i], grays[i + 1]))

    tracks_seq = tracker.track_frames(denoised)

    masks_seq = []
    for f, tracks in zip(denoised, tracks_seq):
        masks_seq.append(segmenter.segment_boxes(f, tracks))

    S_db = compute_mel_spectrogram(audio, sr)
    cwt, _ = compute_cwt_scalogram(audio, sr)
    a_feats = spectral_features(audio, sr)
    try:
        label, prob, topk = audio_model.classify(audio, sr)
    except Exception:
        label, prob, topk = (None, 0.0, [])

    aligned = align_audio_to_frames(frame_ts, audio, sr, window_sec=max(0.25, live_seconds / max(1, len(frames))))
    visual_tokens = []
    audio_tokens = []
    for i, (f, tr, we) in enumerate(zip(denoised, tracks_seq, wavelet_energy)):
        num_boxes = len(tr)
        mean_conf = float(np.mean([t['conf'] for t in tr])) if num_boxes > 0 else 0.0
        vtoken = np.array([num_boxes, mean_conf, we, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        visual_tokens.append(vtoken)
        if len(aligned) > 0:
            aw = aligned[min(i, len(aligned) - 1)]
        else:
            aw = np.zeros(int(0.25 * sr), dtype=np.float32)
        afeats = spectral_features(aw, sr)
        atoken = np.array([
            afeats['zcr'], afeats['centroid'], afeats['bandwidth'], afeats['rolloff'], 0.0, 0.0, 0.0, 0.0
        ], dtype=np.float32)
        audio_tokens.append(atoken)

    if len(visual_tokens) > 0:
        V = torch.from_numpy(np.stack(visual_tokens)[None, ...]).to(cfg['device'])
        A = torch.from_numpy(np.stack(audio_tokens)[None, ...]).to(cfg['device'])
        with torch.no_grad():
            outputs = fusion_head(V, A)
            motion_probs = torch.softmax(outputs['motion'], dim=-1)[0].cpu().numpy()
        moving_prob = float(motion_probs[1])
    else:
        moving_prob = 0.0

    # Dashboard
    st.subheader('Industrial Anomaly Event Log')
    df = pd.DataFrame(event_log)
    st.dataframe(df.tail(10), use_container_width=True)

    # Removed anomaly heatmap display

    st.subheader('Anomaly Trend')
    trend_data = {k: [e['image_anomalies'].count(k) for e in event_log] for k in anomaly_counts}
    st.line_chart(trend_data)

    st.subheader('Anomaly Statistics')
    st.write({k: v for k, v in anomaly_counts.items() if v > 0})

    st.success('Session complete. All events logged.')



if __name__ == '__main__':
    main()

