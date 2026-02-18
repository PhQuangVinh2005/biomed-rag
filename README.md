This repo is created for the Final Project Presentation of the NLP course in USTH

## Members:

- Pham Quang Vinh
- Hoang Khanh Dong
- Nguyen Lam Tung
- Pham Duy Anh
- Nguyen Vu Hong Ngoc
- Le Chi Thanh Lam

## Project Structure

```bash
nlp-project/
├── module/       
│   ├── RAG_pipeline/         # RAG components (chunking, embeddings, retrieval, generation)
│   └── data_processing/  
│       ├── bc5cdr.py         # BC5CDR parser
│       ├── ctd.py            # CTD parser
│       └── pubtator.py       # PubTator/ChemDisGene parser
├── notebooks/      
│   ├── processing_demo/  
│       ├── bc5cdr_processing.ipynb
│       └── chemdisgene_processing.ipynb
│   └── rag_demo/       
│       ├── retrieve_test_result.ipynb
│       └── sample_rag_run.ipynb
├── experiments/              # Experiments
├── finetune/                 # Fine-tuning
├── set_up_dataset.py         # Script to download datasets
├── main.py  
├── requirements.txt  
└── plan.md     
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/PhQuangVinh2005/nlp-project.git
cd nlp-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the template and fill in your details (credentials for Google Sheets and local environment variables).

```bash
cp .env.example .env
```

### 4. Data Preparation

Run the setup script to download and structure the dataset from HuggingFace (`zinzinmit/MedNLPCombined`). This will download BC5CDR, ChemDisGene, and CTD datasets into the `data/external` directory.

Note that the HF_Token can be found in the .env file

```bash
$env:HUGGINGFACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxx"

python set_up_dataset.py
```

---

## Usage & Tutorials

### Data Processing Tutorials

Explore how to load and process the biomedical datasets in the `notebooks/processing_demo/` directory:

- **BC5CDR**: `notebooks/processing_demo/bc5cdr_processing.ipynb`
- **ChemDisGene**: `notebooks/processing_demo/chemdisgene_processing.ipynb`

### RAG Pipeline Demos

See the RAG pipeline in action in the `notebooks/rag_demo/` directory:

- **Pipeline Retrieval Test**: `notebooks/rag_demo/retrieve_test_result.ipynb`
- **Full RAG Run**: `notebooks/rag_demo/sample_rag_run.ipynb`
