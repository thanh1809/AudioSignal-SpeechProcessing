"""Offline Vietnamese neural TTS adapter for Meta MMS VITS."""
from __future__ import annotations
import time
import numpy as np
MODEL_ID='facebook/mms-tts-vie'

def load_backend(local_files_only=True):
    import torch
    from transformers import AutoTokenizer,VitsModel
    torch.set_num_threads(min(4,torch.get_num_threads()))
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,local_files_only=local_files_only)
    model=VitsModel.from_pretrained(MODEL_ID,local_files_only=local_files_only)
    model.eval()
    return model,tokenizer

def synthesize(text,model,tokenizer,seed=2026):
    """Return mono float32 waveform, sampling rate and inference seconds."""
    import torch
    torch.manual_seed(int(seed))
    start=time.perf_counter()
    inputs=tokenizer(text,return_tensors='pt')
    with torch.inference_mode():
        audio=model(**inputs).waveform.squeeze().cpu().numpy()
    elapsed=time.perf_counter()-start
    audio=np.asarray(audio,dtype=np.float32)
    if audio.ndim!=1 or not np.isfinite(audio).all():raise ValueError('TTS produced invalid waveform')
    peak=float(np.max(np.abs(audio))) if audio.size else 0.
    if peak>0:audio=audio*(.95/peak)
    return audio,int(model.config.sampling_rate),float(elapsed)

