import os
import zipfile
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = os.getcwd()

if not BASE_DIR.endswith("biomed-rag"):
    raise ValueError("Please run this script from the nlp-project directory")

data_dir = os.path.join(BASE_DIR, "data")

os.makedirs(data_dir, exist_ok=True)

external_data_dir = os.path.join(data_dir, "external")
os.makedirs(external_data_dir, exist_ok=True)

vectorstore_data_dir = os.path.join(data_dir, "vectorstore")
os.makedirs(vectorstore_data_dir, exist_ok=True)

snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns="bc5cdr/**",
    local_dir=external_data_dir
)

snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns="ChemDisGene/**",
    local_dir=external_data_dir
)

snapshot_download(
    repo_id="bigbio/pubmed_qa",
    repo_type="dataset",
    local_dir=os.path.join(external_data_dir, "pubmed_qa")
)

snapshot_download(
    repo_id="bigbio/med_qa",
    repo_type="dataset",
    local_dir=os.path.join(external_data_dir, "med_qa")
)

# --- Unzip datasets ---

# Unzip med_qa
med_qa_dir = os.path.join(external_data_dir, "med_qa")
med_qa_zip = os.path.join(med_qa_dir, "data_clean.zip")
if os.path.exists(med_qa_zip):
    print("Extracting med_qa/data_clean.zip...")
    with zipfile.ZipFile(med_qa_zip, "r") as z:
        z.extractall(med_qa_dir)
    print("Done.")

# Unzip pubmed_qa
pubmed_qa_dir = os.path.join(external_data_dir, "pubmed_qa")
for zf in ["pqal.zip", "pqaa.zip", "pqau.zip"]:
    zip_path = os.path.join(pubmed_qa_dir, zf)
    if os.path.exists(zip_path):
        print(f"Extracting pubmed_qa/{zf}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(pubmed_qa_dir)
        print("Done.")
