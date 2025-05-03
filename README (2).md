# NLP-Final-project-Task1
# Software Name Recognition in Biomedical Text using BiLSTM-CRF

This project implements a sequence-labeling approach to recognize software names in biomedical text using a bidirectional LSTM (BiLSTM) with a Conditional Random Field (CRF) layer.

## 📖 Approach Overview
- **BIO tagging**: Label tokens as **B-Software** (beginning), **I-Software** (inside), or **O** (outside) based on annotation spans.
- **BiLSTM**: Captures left and right context for each token.
- **CRF layer**: Enforces valid tag transitions for coherent output.
- **Char-level embeddings**: A character-level LSTM helps generalize to rare or novel software names.
- **Training from scratch**: All embeddings and model parameters are learned on your annotated data.

## ⚙️ Prerequisites
- Python 3.8+
- PyTorch
- spaCy
- NumPy

```bash
pip install torch torchvision spacy numpy
python -m spacy download en_core_web_sm
```

## 🔧 Scripts

This repository provides two main Python scripts. Adjust the directory variables at the top of each script before running.

1. **final_project_testdata.py**
   - **Purpose**: Train and evaluate the BiLSTM-CRF model using the **instructor-provided dataset** only.
   - **Input**:
     - `DATA_DIR`: Directory containing the instructor’s `.txt` and `.ann` files (e.g., `data/instructor/`).
   - **Output**:
     - Predicted annotation files (`.ann`) are written to `OUTPUT_DIR` (e.g., `Predicted_annotations/`).

2. **project_newdata.py**
   - **Purpose**: Train the model on the **SoMeSci** (Software Mentions in Science) corpus and **test** on the **instructor-provided dataset**.
   - **Input**:
     - `TRAIN_DIR`: Directory with SoMeSci `.txt` and `.ann` for training.
     - `TEST_DIR`: Directory with the instructor’s `.txt` and `.ann` for testing.
   - **Output**:
     - Predicted `.ann` files for the instructor test set are saved in `OUTPUT_DIR`.

## 🔍 Usage Examples

```bash
# 1. Train & test on instructor dataset
python final_project_testdata.py

# 2. Train on SoMeSci, test on instructor dataset
python project_newdata.py
```

## 📂 Directory Structure
```
.
├── README.md
├── requirements.txt
├── final_project_testdata.py    # Train/test on instructor dataset
├── project_newdata.py          # Train on SoMeSci, test on instructor dataset
├── data/
│   ├── instructor/            # Instructor-provided .txt and .ann files
│   └── SoMeSci/                # SoMeSci corpus .txt and .ann files
└── Predicted_annotations/      # Output directory for all predicted .ann files
```

## 📦 Pre-trained Models
If you already have a trained model (`best_model.pt`), place it in `models/` and skip training—use the appropriate script’s inference step.

## 📝 References
- Michhar, M. (2017). *Named Entity Recognition using a Bi-LSTM with the Conditional Random Field Algorithm*. Data Science <3 Machine Learning. Retrieved from https://michhar.github.io/bilstm-crf-this-is-mind-bending/
- Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional random fields: Probabilistic models for segmenting and labeling sequence data*. In P. Langley (Ed.), Proceedings of the Eighteenth International Conference on Machine Learning (pp. 282–289). Morgan Kaufmann.
- Dang, T. H., Le, H.-Q., Nguyen, T. M., & Vu, S. T. (2018). *D3NER: Biomedical named entity recognition using CRF-biLSTM improved with fine-tuned embeddings of various linguistic information*. Bioinformatics, 34(20), 3539–3546. https://doi.org/10.1093/bioinformatics/bty356
