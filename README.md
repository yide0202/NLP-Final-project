# NLP-Final-project-Task1
# Software Name Recognition in Biomedical Text using BiLSTM-CRF

This project implements a sequence-labeling approach to recognize software names in PubMed abstracts.  
Tokens are labeled with **B-Software** (beginning of a software name), **I-Software** (inside), or **O** (outside). We use a bidirectional LSTM to capture context and a CRF layer to enforce valid tag transitions.

## 📖 Approach Overview
- **BIO tagging**: Each token is assigned `B-Software`, `I-Software`, or `O` based on annotation spans :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.  
- **BiLSTM**: Captures left- and right-context for each token.  
- **CRF layer**: Models dependencies between adjacent tags for coherent output :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}.  
- **Char-level embeddings**: A small LSTM over characters helps the model generalize to rare software names.  
- **Training from scratch**: No pre-trained language models—everything is learned on your annotated data.

## ⚙️ Prerequisites
- Python 3.8+  
- PyTorch  
- spaCy  
- NumPy  

```bash
pip install torch torchvision spacy numpy
python -m spacy download en_core_web_sm
```
## 📂 Directory Structure

```plaintext
.
├── README.md
├── requirements.txt
├── train.py          # Main training script
├── predict.py        # Script to run inference
├── data/
│   ├── train/        # .txt and .ann for training
│   └── test/         # .txt and .ann for testing
└── models/
    └── best_model.pt # Saved BiLSTM-CRF model (after training)
```
## 🔧 Training

1. In `train.py`, set:
   - `TRAIN_DIR` (path to your training `.txt`/`.ann` files)  
   - Hyperparameters (e.g., `num_epochs`, learning rate)

2. Run:
```bash
python train.py
```
   

This will:

Parse Brat annotations

Build vocabularies for words and characters

Train the BiLSTM-CRF model (10 epochs by default)

Save the best model to models/best_model.pt

Log per-epoch loss at the DEBUG level

## 🔍 Evaluation & Inference

After training, run:
```bash
python predict.py \
  --model_path models/best_model.pt \
  --input_dir data/test \
  --output_dir data/predictions
```
This will:

Load your trained model

Predict software name spans on each .txt in data/test

Write .ann files in data/predictions

A small evaluation block at the end computes Precision, Recall, and F1 on your test set.
## 📦 Pre-trained Models
If you have a trained model ready, place it in models/ as best_model.pt and skip training—just use predict.py.

📝 References

- Michhar, M. (2017). *Named Entity Recognition using a Bi-LSTM with the Conditional Random Field Algorithm*. Data Science <3 Machine Learning. Retrieved from https://michhar.github.io/bilstm-crf-this-is-mind-bending/ :contentReference[oaicite:0]{index=0}

- Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional random fields: Probabilistic models for segmenting and labeling sequence data*. In P. Langley (Ed.), Proceedings of the Eighteenth International Conference on Machine Learning (pp. 282–289). Morgan Kaufmann. :contentReference[oaicite:1]{index=1}

- Dang, T. H., Le, H.-Q., Nguyen, T. M., & Vu, S. T. (2018). *D3NER: Biomedical named entity recognition using CRF-biLSTM improved with fine-tuned embeddings of various linguistic information*. Bioinformatics, 34(20), 3539–3546. https://doi.org/10.1093/bioinformatics/bty356 :contentReference[oaicite:2]{index=2}
