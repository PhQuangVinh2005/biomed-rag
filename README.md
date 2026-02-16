This repo is created for the Final Project Presentation of the NLP course in USTH

Members:

- Pham Quang Vinh
- Hoang Khanh Dong
- Nguyen Lam Tung
- Pham Duy Anh
- Nguyen Vu Hong Ngoc
- Le Chi Thanh Lam


## Project Structure

```

biomedical-rag/

├── config.py                 # Model, paths, entity types

├── preprocess.py             # Multi-source → unified docs

├── index.py                  # Build LightRAG KG + vectors

├── query.py                  # QA interface (CLI)

├── evaluate.py               # RAGAS + CID gold evaluation

├── finetune/                 # (Optional) QLoRA fine-tuning

│   ├── prepare_data.py       # BC5CDR → instruction-tuning format

│   ├── train_qlora.py        # QLoRA training script

│   └── export_ollama.py      # Convert to GGUF → Ollama

├── experiments/              # Ablation & comparison results

│   ├── run_ablation.py       # Compare modes, data sources

│   └── results/              # Saved metrics & plots

├── data/

│   ├── CDR_Data/             # BC5CDR (existing)

│   ├── chemdis_gene/         # ChemDisGene

│   ├── ctd/                  # CTD exports

│   └── processed/            # Unified text docs

├── requirements.txt

└── notebooks/

    └── demo.ipynb

```

---


## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/GinHikat/https://github.com/PhQuangVinh2005/nlp-project.git
cd nlp-project
```

### 2. Install Dependencies

```bash
pip install -r requirement.txt
```

### 3. Environment Configuration

Copy the template and fill in your details (credentials for Google Sheets and local environment variables).

```bash
cp .env.example .env
```

### 4. Data Preparation

Run the setup script to download and structure the dataset:

```bash
python set_up_dataset.py
```

---
