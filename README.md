# Language Identification and Diarization

A pipeline for Voice Activity Detection (VAD), Language Identification (LID), and Diarization error evaluation on conversational code-switching speech (English / Mandarin), developed for the MERLIon-CCS Challenge dataset.

---

## 📌 Overview

This project provides an end-to-end workflow for detecting speech segments, classifying spoken languages (English vs. Mandarin), and evaluating diarization and language error rates:

1. **Model Fine-Tuning (`fine_tuning.py`)**: Fine-tunes a pretrained HuBERT Large model (`facebook/hubert-large-ls960-ft`) using a two-phase freeze/unfreeze schedule for sequence classification.
2. **Alternative Training Baseline (`running_model.py`)**: Wav2Vec2-based classification pipeline with dataset sampling and evaluation metrics (ROC, AUC, EER).
3. **Voice Activity Detection (`DD/VAD.ipynb`)**: Segments continuous audio into speech intervals using `pyannote/voice-activity-detection`.
4. **Language Identification (`DD/LI.ipynb`)**: Runs the fine-tuned HuBERT classifier over VAD segments to assign language tags (`English` vs. `Mandarin`).
5. **Evaluation (`DD/error_LI.ipynb`)**: Evaluates predicted language segments against ground-truth intervals using Language Diarization Error Rate (LDER).

---

## 📁 Repository Structure

```
├── DD/
│   ├── VAD.ipynb           # Stage 1: Voice Activity Detection (pyannote.audio)
│   ├── LI.ipynb            # Stage 2: Segment classification with fine-tuned HuBERT
│   ├── error_LI.ipynb      # Stage 3: LDER metric evaluation vs ground truth
│   ├── error_VAD.ipynb     # VAD evaluation notebook (reference)
│   └── error_VAD.py        # VAD evaluation script (reference)
├── labels/
│   └── _MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv  # Ground truth annotations
├── fine_tuning.py          # HuBERT classifier training (Two-phase transfer learning)
├── running_model.py        # Wav2Vec2 training pipeline & evaluation baseline
├── requirements.txt        # Verified Python dependencies
├── .gitignore              # Ignores virtual environments, checkpoints, raw audio
└── README.md
```

---

## ⚙️ Prerequisites & Environment Setup

### 1. Python Environment

Python **3.10** or **3.11** is recommended. On Windows, use a short path for your virtual environment (e.g. `C:\venv\langid`) to avoid the Windows 260-character path limit (`MAX_PATH`):

```powershell
# Create virtual environment at a short path
python -m venv C:\venv\langid
C:\venv\langid\Scripts\activate
```

### 2. Install PyTorch & Dependencies

Install PyTorch first, followed by the remaining requirements:

```powershell
# For CPU inference / testing:
pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# For NVIDIA GPU training (recommended for HuBERT fine-tuning):
# pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu121

# Install project requirements:
pip install -r requirements.txt
```

### 3. Register Jupyter Kernel

To run the notebooks in `DD/`:

```powershell
python -m ipykernel install --user --name langid --display-name "Python (langid)"
```
When opening notebooks in Jupyter or VS Code, select the **Python (langid)** kernel.

---

## 🔑 Hugging Face Authentication (Required for VAD)

The VAD notebook uses `pyannote/voice-activity-detection`, which is a gated Hugging Face model:

1. Visit [huggingface.co/pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection) and accept the user agreement.
2. Visit [huggingface.co/pyannote/segmentation](https://huggingface.co/pyannote/segmentation) and accept the agreement.
3. Generate an Access Token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. Set your token as an environment variable before launching Jupyter:

```powershell
# Windows PowerShell
$env:HF_TOKEN = "your_hf_token_here"

# Linux / macOS
export HF_TOKEN="your_hf_token_here"
```

The notebook automatically reads `os.environ["HF_TOKEN"]`.

---

## 📊 Dataset Preparation

The project is structured around the **MERLIon-CCS Challenge Development Set**:

1. Place all uncompressed `.wav` audio files into an `_audio/` folder at the project root:
   ```
   Language-Identification-and-Diarization/
   ├── _audio/
   │   ├── audio_001.wav
   │   ├── audio_002.wav
   │   └── ...
   ```
2. The ground truth annotations are pre-loaded in `labels/_MERLIon-CCS-Challenge_Development-Set_Language-Labels_v001.csv`.
3. The dataset includes `English`, `Mandarin`, `NON_SPEECH`, and `NON_EVAL_LANG` tags. The sampling pipeline automatically filters for valid speech segments (`English` and `Mandarin` only) with balanced stratified sampling.

---

## 🚀 Execution Workflow

### Stage 1: Train the Classifier
Run the two-phase HuBERT training script:
```powershell
python fine_tuning.py
```
* Phase 1: Freezes the HuBERT backbone, trains the classification head.
* Phase 2: Unfreezes the top layers for domain-specific fine-tuning.
* Checkpoints and training plots are saved to `fine_tune_Results/`.

### Stage 2: Voice Activity Detection
Open and execute `DD/VAD.ipynb`:
* Runs the pyannote VAD pipeline on all audio files in `_audio/`.
* Outputs detected speech intervals to `labels/vad_all_segments_all_audios.csv`.

### Stage 3: Language Identification
Open and execute `DD/LI.ipynb`:
* Loads `fine_tune_Results/phase2_fine_tuned_model.pt`.
* Classifies each VAD interval as English or Mandarin.
* Exports classified intervals to `labels/classified_segments_all.csv`.

### Stage 4: Diarization Error Evaluation
Open and execute `DD/error_LI.ipynb`:
* Compares `labels/classified_segments_all.csv` against the ground truth labels.
* Computes the Language Diarization Error Rate (LDER).

---

## 🛡️ License & Acknowledgements

* **MERLIon-CCS Challenge**: Multilingual and code-switching speech evaluation data.
* **Hugging Face Transformers**: Wav2Vec2 & HuBERT implementations.
* **pyannote.audio**: Neural building blocks for speaker diarization and voice activity detection.
