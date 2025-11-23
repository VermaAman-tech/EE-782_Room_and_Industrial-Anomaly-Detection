import io
import os
import sys
import time
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

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


st.set_page_config(page_title="Multimodal Room Monitor", layout="wide")


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


def overlay_boxes(frame: np.ndarray, boxes: List[dict]) -> np.ndarray:
    vis = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b['xyxy'])
        tid = b.get('track_id', -1)
        label = f"ID {tid}" if tid >= 0 else f"C{b['cls']}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def plot_spectrogram(S_db: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(3, 2))
    im = ax.imshow(S_db, origin='lower', aspect='auto', cmap='magma')
    ax.set_title('Mel-Spectrogram')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return buf.getvalue()


def plot_scalogram(cwt: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(3, 2))
    im = ax.imshow(np.abs(cwt), origin='lower', aspect='auto', cmap='viridis')
    ax.set_title('CWT Scalogram')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return buf.getvalue()


def main():
    cfg = load_config('configs/config.yaml')
    st.title('Multimodal Room Monitor (Open Source, Streamlit)')

    cams = list_cameras()
    devices, mic_names = list_microphones()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        cam_idx = st.selectbox('Camera Index', cams if len(cams) > 0 else [0])
    with col2:
        mic_idx = st.number_input('Microphone Device Index', min_value=0, max_value=max(0, len(mic_names) - 1), value=0, step=1)
    with col3:
        st.write('')

    detector, tracker, segmenter, audio_model, fusion_head = load_components(cfg)

    # Live capture controls
    dur_col1, dur_col2 = st.columns([1, 1])
    with dur_col1:
        live_seconds = st.slider('Live Capture Duration (seconds)', min_value=2, max_value=10, value=int(max(2, int(cfg['audio']['duration_sec']))))
    with dur_col2:
        burst_frames = st.slider('Frames to collect', min_value=cfg['burst']['num_frames'], max_value=max(cfg['burst']['num_frames'], 60), value=cfg['burst']['num_frames'], step=1)

    live_btn = st.button('Start Live Capture + Analysis')

    if live_btn:
        import sounddevice as sd

        st.info('Live capture started. Showing live detections and audio spectrogram...')
        live_video = st.empty()
        live_spec = st.empty()
        live_stats = st.empty()

        cap = cv2.VideoCapture(int(cam_idx))
        if not cap.isOpened():
            st.error('Failed to open camera')
            return

        sr = int(cfg['audio']['sample_rate'])
        channels = int(cfg['audio']['channels'])
        audio_buf = []
        blocksize = 1024
        start_time = time.time()

        def audio_cb(indata, frames, time_info, status):
            if status:
                pass
            audio_buf.append(indata.copy().squeeze(-1) if channels == 1 else indata.copy())

        stream = sd.InputStream(device=int(mic_idx), samplerate=sr, channels=channels, blocksize=blocksize, callback=audio_cb)
        stream.start()

        collected_frames: List[np.ndarray] = []
        frame_ts: List[float] = []
        frame_interval = max(0.001, cfg['burst']['frame_interval_ms'] / 1000.0)
        last_spec_update = 0.0

        try:
            while len(collected_frames) < burst_frames and (time.time() - start_time) < live_seconds:
                ok, frame = cap.read()
                if not ok:
                    continue
                ts = time.time()
                frame_ts.append(ts)
                collected_frames.append(frame)

                # Fast live detection for overlay
                dets = detector.detect([frame])[0]
                vis = overlay_boxes(frame, dets)
                live_video.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True)

                # Periodically update audio spectrogram
                if (ts - last_spec_update) > 0.3 and len(audio_buf) > 0:
                    last_spec_update = ts
                    audio_now = np.concatenate(audio_buf, axis=0)
                    win = int(0.8 * sr)
                    if len(audio_now) > win:
                        audio_now = audio_now[-win:]
                    S_live = compute_mel_spectrogram(audio_now, sr)
                    live_spec.image(plot_spectrogram(S_live), caption='Live Mel-Spectrogram', use_column_width=True)

                num_boxes = len(dets)
                mean_conf = float(np.mean([d['conf'] for d in dets])) if num_boxes > 0 else 0.0
                live_stats.write(f"Frames: {len(collected_frames)}/{burst_frames} | Live detections: {num_boxes} | Mean conf: {mean_conf:.2f}")

                time.sleep(frame_interval)
        finally:
            cap.release()
            stream.stop(); stream.close()

        audio_all = np.concatenate(audio_buf, axis=0) if len(audio_buf) > 0 else np.zeros(int(sr * max(0.1, live_seconds)))

        # Detailed analysis
        st.success('Live capture finished. Running detailed analysis...')

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
        label, prob, topk = audio_model.classify(audio, sr)

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
                logits = fusion_head(V, A)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            moving_prob = float(probs[1])
        else:
            moving_prob = 0.0

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader('Video Analysis (Detailed)')
            for i, (f, tr) in enumerate(zip(denoised, tracks_seq)):
                vis = overlay_boxes(f, tr)
                if i < len(flows):
                    vis = draw_flow_arrows(vis, flows[i])
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f'Frame {i}', use_column_width=True)
        with c2:
            st.subheader('Audio Analysis (Detailed)')
            st.metric('Top Audio Label', f'{label}', f'{prob*100:.1f}%')
            st.image(plot_spectrogram(S_db))
            st.image(plot_scalogram(cwt))
            st.write('Top-5 classes:')
            for name, p in topk:
                st.write(f'- {name}: {p*100:.1f}%')

        st.subheader('Fusion Result')
        st.write(f"Estimated moving activity probability: {moving_prob:.3f}")
        st.progress(moving_prob)

        st.subheader('Detailed Per-frame Summary')
        for i, tr in enumerate(tracks_seq):
            ids = [t['track_id'] for t in tr if t['track_id'] >= 0]
            st.write(f"Frame {i}: {len(tr)} detections; Tracks: {ids}")

    if st.button('Capture Burst', type='primary'):
        with st.spinner('Capturing frames and audio...'):
            frames, frame_ts = capture_burst(int(cam_idx), cfg['burst']['num_frames'], cfg['burst']['frame_interval_ms'])
            audio = record_audio(cfg['audio']['duration_sec'], cfg['audio']['sample_rate'], cfg['audio']['channels'])

        st.success('Captured')

        # Preprocess frames and compute simple features
        denoised = [denoise_frame(f) for f in frames]
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in denoised]
        wavelet_energy = [compute_wavelet_energy(g.astype(np.float32) / 255.0) for g in grays]

        # Optical flow between consecutive frames (TV-L1)
        flows = []
        for i in range(len(grays) - 1):
            flows.append(optical_flow_tvl1(grays[i], grays[i + 1]))

        # Detection + ByteTrack tracking
        tracks_seq = tracker.track_frames(denoised)

        # Segmentation masks per frame
        masks_seq = []
        for f, tracks in zip(denoised, tracks_seq):
            masks_seq.append(segmenter.segment_boxes(f, tracks))

        # Audio features and classification
        S_db = compute_mel_spectrogram(audio, cfg['audio']['sample_rate'])
        cwt, _ = compute_cwt_scalogram(audio, cfg['audio']['sample_rate'])
        a_feats = spectral_features(audio, cfg['audio']['sample_rate'])
        label, prob, topk = audio_model.classify(audio, cfg['audio']['sample_rate'])

        # Align audio windows to frames for fusion (toy stats for demo)
        aligned = align_audio_to_frames(frame_ts, audio, cfg['audio']['sample_rate'], window_sec=cfg['audio']['duration_sec'] / max(1, cfg['burst']['num_frames']))
        # Build small feature tokens: visual [bbox count, mean conf, wavelet] ; audio [zcr, centroid, bandwidth, rolloff]
        visual_tokens = []
        audio_tokens = []
        for i, (f, tr, we) in enumerate(zip(denoised, tracks_seq, wavelet_energy)):
            num_boxes = len(tr)
            mean_conf = float(np.mean([t['conf'] for t in tr])) if num_boxes > 0 else 0.0
            vtoken = np.array([num_boxes, mean_conf, we, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            visual_tokens.append(vtoken)
            aw = aligned[min(i, len(aligned) - 1)]
            afeats = spectral_features(aw, cfg['audio']['sample_rate'])
            atoken = np.array([
                afeats['zcr'], afeats['centroid'], afeats['bandwidth'], afeats['rolloff'], 0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
            audio_tokens.append(atoken)

        V = torch.from_numpy(np.stack(visual_tokens)[None, ...])
        A = torch.from_numpy(np.stack(audio_tokens)[None, ...])
        V = V.to(cfg['device'])
        A = A.to(cfg['device'])
        with torch.no_grad():
            logits = fusion_head(V, A)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        moving_prob = float(probs[1])

        # UI Layout
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader('Video Analysis')
            for i, (f, tr) in enumerate(zip(denoised, tracks_seq)):
                vis = overlay_boxes(f, tr)
                if i < len(flows):
                    vis = draw_flow_arrows(vis, flows[i])
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f'Frame {i}', use_column_width=True)
        with c2:
            st.subheader('Audio Analysis')
            st.metric('Top Audio Label', f'{label}', f'{prob*100:.1f}%')
            st.image(plot_spectrogram(S_db))
            st.image(plot_scalogram(cwt))
            st.write('Top-5 classes:')
            for name, p in topk:
                st.write(f'- {name}: {p*100:.1f}%')

        st.subheader('Fusion Result')
        st.write(f"Estimated moving activity probability: {moving_prob:.3f}")
        st.progress(moving_prob)

        st.subheader('Detailed Per-frame Summary')
        for i, tr in enumerate(tracks_seq):
            ids = [t['track_id'] for t in tr if t['track_id'] >= 0]
            st.write(f"Frame {i}: {len(tr)} detections; Tracks: {ids}")


if __name__ == '__main__':
    main()

