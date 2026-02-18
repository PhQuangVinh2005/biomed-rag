import os
from huggingface_hub import hf_hub_download, snapshot_download

BASE_DIR = os.getcwd()

if not BASE_DIR.endswith("nlp-project"):
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

# snapshot_download(
#     repo_id="zinzinmit/MedNLPCombined",
#     repo_type="dataset",
#     allow_patterns="CTD/**",
#     local_dir=external_data_dir
# )