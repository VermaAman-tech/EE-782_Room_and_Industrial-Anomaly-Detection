from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification,
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC,
    HubertForSequenceClassification
)

class AdvancedAudioAnalyzer:
    def __init__(
        self, 
        model_id: str, 
        device: str = 'cpu',
        wav2vec_model_id: str = 'facebook/wav2vec2-base-960h',
        hubert_model_id: str = 'facebook/hubert-base-ls960'
    ):
        """
        Initialize the advanced audio analyzer.
        Args:
            model_id: AST model ID or path for audio classification
            device: Device to run the models on ('cpu' or 'cuda')
            wav2vec_model_id: Model ID for the Wav2Vec2 speech recognition model
            hubert_model_id: Model ID for the HuBERT acoustic scene understanding model
        """
        self.device = device
        
        # AST for general audio classification
        try:
            self.ast_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.ast_model = AutoModelForAudioClassification.from_pretrained(
                model_id,
                output_hidden_states=True
            ).to(self.device)
            self.ast_model.eval()
        except Exception as e:
            print(f"Warning: Failed to load AST model: {e}")
            self.ast_model = None
        
        # Wav2Vec2 for speech recognition
        try:
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_id)
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(
                wav2vec_model_id,
                output_hidden_states=True
            ).to(self.device)
            self.wav2vec_model.eval()
        except Exception as e:
            print(f"Warning: Failed to load Wav2Vec2 model: {e}")
            self.wav2vec_model = None
        
        # HuBERT for acoustic scene understanding
        try:
            self.hubert_model = HubertForSequenceClassification.from_pretrained(
                hubert_model_id,
                output_hidden_states=True,
                num_labels=527  # AudioSet classes
            ).to(self.device)
            self.hubert_model.eval()
        except Exception as e:
            print(f"Warning: Failed to load HuBERT model: {e}")
            self.hubert_model = None
        
        # Fusion layer for combining embeddings
        self.fusion = nn.Sequential(
            nn.Linear(768 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        ).to(device)

    def _prepare_audio(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Prepare audio by resampling if needed"""
        if sample_rate != 16000:
            from scipy import signal
            samples = int((len(audio) * 16000) / sample_rate)
            audio = signal.resample(audio, samples)
            sample_rate = 16000
        
        # Normalize audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        return audio, sample_rate

    @torch.no_grad()
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Analyze audio using multiple models for comprehensive understanding.
        Args:
            audio: Audio signal as numpy array
            sample_rate: Sampling rate of the audio signal
        Returns:
            Dictionary containing audio events, transcription, embeddings, and top predictions
        """
        # Prepare audio
        audio, sample_rate = self._prepare_audio(audio, sample_rate)
        
        # Initialize outputs
        ast_predictions = []
        transcription = ""
        ast_embeds = torch.zeros(1, 768).to(self.device)
        wav2vec_embeds = torch.zeros(1, 768).to(self.device)
        hubert_embeds = torch.zeros(1, 768).to(self.device)
            
        # 1. AST Analysis for general audio classification
        if self.ast_model is not None:
            try:
                ast_inputs = self.ast_extractor(audio, sampling_rate=sample_rate, return_tensors="pt")
                ast_inputs = {k: v.to(self.device) for k, v in ast_inputs.items()}
                ast_outputs = self.ast_model(**ast_inputs)
                ast_probs = torch.softmax(ast_outputs.logits, dim=-1)[0]
                ast_topk = torch.topk(ast_probs, k=min(5, ast_probs.shape[-1]))
                ast_predictions = [(self.ast_model.config.id2label[int(i)], float(p)) 
                                for p, i in zip(ast_topk.values.tolist(), ast_topk.indices.tolist())]
                ast_embeds = ast_outputs.hidden_states[-1].mean(dim=1)
            except Exception as e:
                print(f"Warning: AST analysis failed: {e}")
        
        # 2. Wav2Vec Analysis for speech content
        if self.wav2vec_model is not None:
            try:
                wav2vec_inputs = self.wav2vec_processor(audio, sampling_rate=sample_rate, return_tensors="pt")
                wav2vec_inputs = {k: v.to(self.device) for k, v in wav2vec_inputs.items()}
                wav2vec_outputs = self.wav2vec_model(**wav2vec_inputs)
                wav2vec_logits = wav2vec_outputs.logits
                wav2vec_tokens = torch.argmax(wav2vec_logits, dim=-1)
                transcription = self.wav2vec_processor.batch_decode(wav2vec_tokens)[0]
                if hasattr(wav2vec_outputs, 'hidden_states'):
                    wav2vec_embeds = wav2vec_outputs.hidden_states[-1].mean(dim=1)
            except Exception as e:
                print(f"Warning: Wav2Vec analysis failed: {e}")
        
        # 3. HuBERT Analysis for acoustic scene
        if self.hubert_model is not None:
            try:
                # Prepare and normalize input values
                input_values = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
                input_values = (input_values - input_values.mean()) / (input_values.std() + 1e-6)
                input_values = input_values.to(self.device)
                
                hubert_outputs = self.hubert_model(input_values)
                if hasattr(hubert_outputs, 'hidden_states'):
                    hubert_embeds = hubert_outputs.hidden_states[-1].mean(dim=1)
                else:
                    hubert_embeds = hubert_outputs.logits.mean(dim=1).unsqueeze(0)
            except Exception as e:
                print(f"Warning: HuBERT analysis failed: {e}")
        
        # 4. Fusion of embeddings for comprehensive understanding
        try:
            combined_embeds = torch.cat([ast_embeds, wav2vec_embeds, hubert_embeds], dim=-1)
            fused_features = self.fusion(combined_embeds)
        except Exception as e:
            print(f"Warning: Fusion failed: {e}")
            fused_features = torch.zeros(1, 256).to(self.device)
        
        # Return comprehensive analysis with defaults if components failed
        return {
            'audio_events': ast_predictions or [("unknown", 0.0)],
            'transcription': transcription or "",
            'acoustic_embeddings': fused_features.cpu().numpy(),
            'top_event': ast_predictions[0][0] if ast_predictions else "unknown",
            'top_confidence': ast_predictions[0][1] if ast_predictions else 0.0
        }

    @torch.no_grad()
    def classify(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Legacy method for compatibility with older code.
        Args:
            audio: Audio signal as numpy array
            sample_rate: Sampling rate of the audio signal
        Returns:
            Tuple of (top event label, confidence score, list of (label, score) pairs)
        """
        results = self.analyze(audio, sample_rate)
        return results['top_event'], results['top_confidence'], results['audio_events']


# For backward compatibility
AudioClassifierAST = AdvancedAudioAnalyzer