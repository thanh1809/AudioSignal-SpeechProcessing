# Lab 5 - Speech enhancement, speaker recognition and separation

This project implements the Lab 5 pipeline end to end: additive noise at controlled SNR, STFT spectral subtraction and Wiener filtering, MFCC speaker identification and verification, and two-channel FastICA separation. It also runs parameter, feature, mixing-matrix, and noise ablations and writes the audio, tables, and figures under `results/`.

## Data and provenance

The `lab5_data/clean/` enrollment, development, and test clips are reproducible speech-synthesis examples. `spk01` and `spk02` were synthesized locally with the installed Microsoft SAPI voices David and Zira. `spk03` is a deterministic pitch-shifted derivative of `spk02` used only as a third toy class; it is not an independent human speaker. Fan-like and cafeteria-like noise are generated deterministically by the script. Separation sources are two of the synthesized utterances. No participant recordings or externally sourced audio are included.

This dataset makes the algorithms runnable without distributing unconsented recordings. Results are a teaching demonstration, not evidence of performance on independent people or real-world noise. For a valid human-speaker evaluation, replace the WAVs with consented data while preserving the train/dev/test naming and split; do not treat the derived `spk03` as an independent speaker.

All input audio is read as mono and resampled to 16 kHz by the pipeline. The script creates any missing deterministic noise and `spk03` examples. The SAPI source WAVs in the clean directory are included so the project runs independently of the system TTS voices.

## Environment

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python lab5.py
```

Open `Lab5.ipynb` and run all cells for the notebook version. The notebook calls the same deterministic pipeline as the script. The run uses seed 457 and records its parameters in `results/tables/run_config.json`.

## Data layout

```text
lab5_data/
  clean/       spk01/spk02 enrollment, dev and test WAVs; spk03 is generated
  noise/       generated fan.wav and cafeteria.wav
  separation/  source_A.wav and source_B.wav
results/
  audio/       noisy/enhanced speech, microphone mixtures and ICA estimates
  figures/     spectrograms, confusion matrix, FAR-FRR, ICA correlation
  tables/      SNR/ablation/trial/ICA metrics and run configuration
```

The verification threshold is selected using only the `dev` utterances and is frozen for `test`. FastICA receives only the two microphone mixtures; the mixing matrix is used to create the known synthetic mixture and to report its determinant, not as an ICA input.

## Main outputs

- `results/audio/noisy_{0,5,10}dB.wav`, `enhanced_ss_{0,5,10}dB.wav`, and `enhanced_wiener_{0,5,10}dB.wav`
- `results/audio/mix_mic1.wav`, `mix_mic2.wav`, `separated_1.wav`, and `separated_2.wav`
- `results/tables/enhancement_snr.csv`, `ss_parameter_sweep.csv`, `speaker_id_scores.csv`, `confusion_matrix.csv`, `speaker_feature_ablation.csv`, `speaker_condition_accuracy.csv`, `verification_threshold_curve.csv`, `verification_test_trials.csv`, `verification_summary.json`, and `ica_conditions.csv`
- `results/figures/enhancement_spectrograms.png`, `confusion_matrix.png`, `far_frr.png`, `ica_correlation.png`, and `ica_waveforms_spectrograms.png`

The report is `report.docx`. Regenerate the report after a new run with `python build_report.py`.
