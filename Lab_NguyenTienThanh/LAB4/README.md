# Lab 4 Mini TTS

## Run

Install the Python dependencies, prepare the Vietnamese checkpoint once, then run the full pipeline:

```powershell
python -m pip install -r requirements.txt
python prepare_neural_tts.py
python lab4.py
```

`prepare_neural_tts.py` downloads only the documented Meta MMS Vietnamese VITS checkpoint into the Hugging Face cache. After that one-time step, `lab4.py` loads it with `local_files_only=True` and does not access the network. The model is CC-BY-NC 4.0; use it only in non-commercial contexts and retain attribution. We do not commit model weights.

You can also run `Lab04.ipynb` from this directory. Outputs are written to `audio_output/`, `lab4_data/`, and `results/`.

## Inventory provenance

The Lab 2 repository has seven recorded command labels with five copies per label. The original `dataset/` files are byte-identical within each label, so the pipeline takes five distinct augmented recordings per label from `../LAB2/Lab2_data_figures/dataset_aug/`. Those are augmentation-derived candidates, not five independent natural takes.

To complete the Lab 4 vocabulary, the pipeline uses the seven real command labels plus nine word units (`mời`, `khách`, `hàng`, `số`, `hai`, `ba`, `đến`, `quầy`, `năm`) generated in two seeded variants each by the offline Vietnamese VITS model. The resulting inventory has 16 distinct labels and 53 audio candidates. Metadata identifies which units are recordings and which are model-generated. The exact queue sentence “Mời khách hàng số 23 đến quầy 5.” is normalized and synthesized through concatenative, HMM-state hybrid, and neural VITS backends.

## Scope of the HMM and listening evaluation

The HMM demonstration estimates five state-level F0, RMS, spectral-centroid and duration parameters per word. Its state duration controls time-stretch of the source units; `hmm_tts.wav` is an educational HMM-state-controlled hybrid, not a separately trained parametric voice plus vocoder. The state trajectory and waveform are included for inspection.

`results/ratings.csv` is a blank listening-test form for five listeners and four utterances. MOS, naturalness and human intelligibility scores are not fabricated; collect those ratings before making subjective quality claims. Automatic token-to-unit coverage and boundary measures are provided as objective diagnostics.

## Main artifacts

- `lab4.py`, `Lab04.ipynb`: runnable pipeline.
- `prepare_neural_tts.py`, `neural_tts_backend.py`: one-time model preparation and local-only inference adapter.
- `lab4_data/metadata.csv`, `lab4_data/units/`: normalized mono 16 kHz inventory with source provenance.
- `lab4_data/hmm_demo/state_parameters.csv`: state-wise acoustic/prosody parameters.
- `audio_output/`: raw/best concatenation, queue HMM hybrid, neural VITS audio, three additional test sentences, cross-fade variants and pitch/speed trials.
- `results/`: DP/join costs, metrics, inference latencies, mel-spectrogram, figures and the blank listening form.
- `Bao_cao_Lab4.pdf` and `Bao_cao_Lab4.docx`: 5-8 page report covering methods, measurements and all analysis questions.
