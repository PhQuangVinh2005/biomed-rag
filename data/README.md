# Datasets

This directory contains two biomedical QA datasets used for the BioMed-RAG project.

## Directory Structure

```
data/
├── external/
│   ├── pubmed_qa/          # PubMedQA dataset (bigbio/pubmed_qa)
│   │   ├── pqal/           # Labeled subset (expert-annotated)
│   │   │   ├── train_set.json
│   │   │   └── dev_set.json
│   │   ├── pqaa/           # Artificial subset (auto-generated)
│   │   │   ├── train_set.json
│   │   │   └── dev_set.json
│   │   └── pqau/           # Unlabeled subset
│   │       └── pqau.json
│   └── med_qa/             # MedQA dataset (bigbio/med_qa)
│       └── data_clean/
│           ├── questions/
│           │   ├── US/             # English (USMLE)
│           │   ├── Mainland/       # Simplified Chinese
│           │   └── Taiwan/         # Traditional Chinese
│           └── textbooks/
│               ├── en/             # English medical textbooks
│               ├── zh_paragraph/   # Chinese (paragraph-split)
│               └── zh_sentence/    # Chinese (sentence-split)
└── vectorstore/            # Vector store for RAG
```

---

## 1. PubMedQA

> **Source**: [bigbio/pubmed_qa](https://huggingface.co/datasets/bigbio/pubmed_qa)
> **License**: MIT
> **Language**: English
> **Task**: Yes/No/Maybe Question Answering

### Description

PubMedQA is a biomedical QA dataset collected from PubMed abstracts. The task is to answer research questions with **yes/no/maybe** using the corresponding abstracts. It is the first QA dataset requiring reasoning over biomedical research texts, especially their quantitative contents.

### Instance Structure

Each sample contains:

| Field | Description |
|-------|-------------|
| `QUESTION` | Research question (from article title) |
| `CONTEXTS` | PubMed abstract paragraphs (without conclusion) |
| `LABELS` | Section labels for each context paragraph |
| `MESHES` | MeSH terms associated with the article |
| `LONG_ANSWER` | Conclusion of the abstract |
| `final_decision` | yes / no / maybe |

### Subsets

| Subset | Files | Samples | Description |
|--------|-------|---------|-------------|
| **PQA-L** (Labeled) | `pqal/train_set.json`, `dev_set.json` | ~1,000 | Expert-annotated yes/no/maybe QA |
| **PQA-A** (Artificial) | `pqaa/train_set.json`, `dev_set.json` | ~211,300 | Auto-generated questions with heuristic yes/no labels |
| **PQA-U** (Unlabeled) | `pqau/pqau.json` | ~61,200 | Context-question pairs without answers |

### Citation

```bibtex
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing
             and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}
```

---

## 2. MedQA

> **Source**: [bigbio/med_qa](https://huggingface.co/datasets/bigbio/med_qa)
> **License**: Unknown
> **Languages**: English, Simplified Chinese, Traditional Chinese
> **Task**: Multiple-Choice Question Answering

### Description

MedQA is the first free-form multiple-choice OpenQA dataset for solving medical problems, collected from professional medical board exams (USMLE for English, national exams for Chinese). It also includes a large-scale corpus of medical textbooks for retrieval-based approaches.

### Instance Structure (JSONL)

| Field | Description |
|-------|-------------|
| `question` | Medical exam question |
| `options` | Dict of answer choices (e.g., `{"A": "...", "B": "...", ...}`) |
| `answer_idx` | Correct answer key (e.g., `"A"`) |
| `answer` | Full text of the correct answer |
| `meta_info` | Source metadata |

### Question Splits

| Language | Train | Dev | Test | Total |
|----------|-------|-----|------|-------|
| **English (US)** | ~10,178 | ~1,272 | ~1,273 | **12,723** |
| **Simplified Chinese (Mainland)** | ~27,400 | ~3,425 | ~3,426 | **34,251** |
| **Traditional Chinese (Taiwan)** | ~11,298 | ~1,412 | ~1,413 | **14,123** |

Additionally, there are **4-option variants** for English and Chinese subsets, and **translated versions** (Taiwan→English, Taiwan→Chinese) under the Taiwan directory.

### Textbook Corpus

The dataset includes a medical textbook corpus useful as a **retrieval knowledge base for RAG**:

| Language | Contents |
|----------|----------|
| **English** (`textbooks/en/`) | 18 medical textbooks covering Anatomy, Physiology, Pathology, Pharmacology, Biochemistry, Surgery, etc. |
| **Chinese** (`textbooks/zh_paragraph/`, `zh_sentence/`) | Chinese medical textbooks in paragraph-split and sentence-split formats |

### Citation

```bibtex
@article{jin2021disease,
  title={What disease does this patient have? A large-scale open domain question answering
         dataset from medical exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung
          and Fang, Hanyi and Szolovits, Peter},
  journal={Applied Sciences},
  volume={11},
  number={14},
  pages={6421},
  year={2021},
  publisher={MDPI}
}
```

---

## Comparison

| | PubMedQA | MedQA |
|---|---|---|
| **Domain** | Research abstracts | Medical board exams |
| **Question Type** | Yes/No/Maybe | Multiple-choice (4–5 options) |
| **Context** | PubMed abstract | Exam question stem |
| **Total Samples** | ~273,500 | ~61,097 |
| **Languages** | English | English, Chinese (Simplified & Traditional) |
| **Retrieval Corpus** | Abstract contexts (built-in) | Medical textbooks (separate) |
| **RAG Use Case** | Answer biomedical research questions using abstract reasoning | Answer clinical exam questions using textbook knowledge |
