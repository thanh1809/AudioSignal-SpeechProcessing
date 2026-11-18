# Lab 3 Gaussian HMM Speech Recognition

## Run
Use Python 3.10+ and install the packages used by this implementation:

```powershell
python -m pip install numpy scipy pandas librosa matplotlib seaborn scikit-learn jupyter
jupyter notebook Lab03.ipynb
```

Alternatively run `python lab3.py` from this directory. The implementation uses NumPy/SciPy for Gaussian HMM Forward-Backward, Baum-Welch parameter updates, likelihood and Viterbi, so `hmmlearn` is not required. Processing 25 WAV files and producing all figures takes roughly one minute on the reference machine.

## Data and split
Input recordings are reused from `../LAB2/Lab2_data_figures/dataset`. This lab selects five classes: `bat`, `dung`, `len`, `xuong`, `trai`. Files 01-03 per class are training; 04-05 are held-out test. No file appears in both splits. The available set has only five utterances per class, rather than the 8 suggested in the PDF. `results/split.json` records the split.

## Contents
- `lab3.py`: feature extraction, CMVN, left-to-right diagonal Gaussian HMM with Baum-Welch, DTW baseline, state-size/static-dynamic/noise experiments, Viterbi and bigram re-ranking.
- `Lab03.ipynb`: lab narrative and one-cell end-to-end runner.
- `models/`: five fitted five-state HMM parameter archives (`.npz`).
- `results/`: score tables, confusion matrices, metrics, robustness, state experiment, LM sweep and Viterbi paths.
- `results/figures/`: two dynamic feature heatmaps, ten Viterbi alignments and two confusion matrices.
- `lexicon.txt`: example WORD-to-phone-sequence file. Its phones are illustrative and must be replaced with the instructor's inventory.
- `command_corpus.txt`: small illustrative command sequence corpus for unigram/bigram counts; not a supplied conversational corpus.
- `Bao_cao_Lab3.docx`: concise report, method, results, lab questions and limitations.

## Key implementation details
Sampling rate is 16 kHz, mono, with silence trimming, 0.97 pre-emphasis, 13 MFCCs at 25 ms / 10 ms hop, delta and delta-delta (39 dimensions), and utterance CMVN. Whole-word HMMs use 5 states, diagonal covariance and a Bakis transition topology. The 3/5/7 state sweep and a 13D static-MFCC HMM comparison are included. DTW uses the same train/test files and three training templates per class.

The robustness trial adds white noise at 20 dB SNR to held-out recordings. This is a controlled noise test, not a new-speaker evaluation. The phone sequences and corpus exist only to demonstrate the required file format and add-k re-ranking; replace them with approved class material for a final course submission. Full covariance is not fitted because three utterances per class are too few for stable 39D estimates.
