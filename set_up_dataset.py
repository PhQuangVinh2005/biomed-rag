import json
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

# --- Config ---
BASE_DIR = Path(os.getcwd())
assert BASE_DIR.name == "biomed-rag", "Please run this script from the biomed-rag directory"

DATA_DIR = BASE_DIR / "data"
EXTERNAL_DIR = DATA_DIR / "external"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

for d in [DATA_DIR, EXTERNAL_DIR, VECTORSTORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- 1. Download datasets ---
DATASETS = {
    "bigbio/pubmed_qa": EXTERNAL_DIR / "pubmed_qa",
    "bigbio/med_qa": EXTERNAL_DIR / "med_qa",
}

for repo_id, local_dir in DATASETS.items():
    print(f"\nDownloading {repo_id}...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local_dir))

# --- 2. Extract zip files ---
med_qa_dir = EXTERNAL_DIR / "med_qa"
pubmed_qa_dir = EXTERNAL_DIR / "pubmed_qa"

zips_to_extract = [
    (med_qa_dir / "data_clean.zip", med_qa_dir),
    (pubmed_qa_dir / "pqal.zip", pubmed_qa_dir),
    (pubmed_qa_dir / "pqaa.zip", pubmed_qa_dir),
    (pubmed_qa_dir / "pqau.zip", pubmed_qa_dir),
]

for zip_path, dest in zips_to_extract:
    if zip_path.exists():
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest)
        zip_path.unlink()
        print(f"  Extracted & deleted {zip_path.name}")

# --- 3. Cleanup redundant files ---
for name in ["pqal_test_set.json", "pqal_train_dev_set.json"]:
    p = pubmed_qa_dir / name
    if p.exists():
        p.unlink()
        print(f"Deleted {name}")

# --- 4. Merge pqal folds into single pqal/ directory ---
pqal_dir = pubmed_qa_dir / "pqal"
pqal_dir.mkdir(exist_ok=True)

merged = {"train_set.json": {}, "dev_set.json": {}}

for i in range(10):
    fold = pubmed_qa_dir / f"pqal_fold{i}"
    if not fold.is_dir():
        continue
    for filename in merged:
        fp = fold / filename
        if fp.exists():
            merged[filename].update(json.loads(fp.read_text()))
    shutil.rmtree(fold)

for filename, data in merged.items():
    (pqal_dir / filename).write_text(json.dumps(data, indent=2))
    print(f"pqal/{filename}: {len(data)} samples")

# --- 5. Reorganize pqaa & pqau into subdirectories ---
file_moves = [
    ("pqaa_train_set.json", "pqaa/train_set.json"),
    ("pqaa_dev_set.json", "pqaa/dev_set.json"),
    ("ori_pqau.json", "pqau/pqau.json"),
]

for old, new in file_moves:
    src = pubmed_qa_dir / old
    dst = pubmed_qa_dir / new
    if src.exists():
        dst.parent.mkdir(exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"Moved {old} -> {new}")
