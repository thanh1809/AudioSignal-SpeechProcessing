"""One-time checkpoint preparation. Inference thereafter runs local-only."""
from transformers import AutoTokenizer,VitsModel

MODEL_ID='facebook/mms-tts-vie'
if __name__=='__main__':
    AutoTokenizer.from_pretrained(MODEL_ID)
    model=VitsModel.from_pretrained(MODEL_ID)
    print(f'Cached {MODEL_ID}; sampling rate {model.config.sampling_rate} Hz; parameters {sum(p.numel() for p in model.parameters()):,}')
    print('License: CC-BY-NC 4.0 (non-commercial). Check the model card before other uses.')
