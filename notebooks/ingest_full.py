"""
ingest_full.py — Ingest all documents into LightRAG.

Sources (in order, smallest→largest):
  medqa/textbooks/  — 18 medical textbooks (.txt)
  pubmedqa/         — PubMedQA abstracts (.csv, abstract column)

Usage:
    python notebooks/ingest_full.py              # ingest everything
    python notebooks/ingest_full.py --textbooks  # textbooks only
    python notebooks/ingest_full.py --pubmedqa   # pubmedqa only
    python notebooks/ingest_full.py --dry-run    # list files without ingesting
"""
import asyncio
import sys
import os
import argparse

# Ensure rag_config is importable regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rag_config import (
    build_rag, project_root,
    LLM_MODEL, WORKING_DIR, LLM_MAX_TOKENS, DEBUG_LLM, DEBUG_OUTPUT_FILE,
)

# ── Document sources ──────────────────────────────────────────────────────────
TEXTBOOK_DIR = os.path.join(project_root, "data", "external", "medqa", "textbooks")
PUBMEDQA_CSV = os.path.join(project_root, "data", "external", "pubmedqa", "pubmedqa.csv")

# Textbooks ordered smallest → largest to front-load faster wins
TEXTBOOKS = [
    "Pathoma_Husain.txt",
    "First_Aid_Step1.txt",
    "First_Aid_Step2.txt",
    "Biochemistry_Lippincott.txt",
    "Anatomy_Gray.txt",
    "Psichiatry_DSM-5.txt",
    "Pediatrics_Nelson.txt",
    "Physiology_Levy.txt",
    "Histology_Ross.txt",
    "Immunology_Janeway.txt",
    "Pathology_Robbins.txt",
    "Cell_Biology_Alberts.txt",
    "Pharmacology_Katzung.txt",
    "Gynecology_Novak.txt",
    "Obstentrics_Williams.txt",
    "Neurology_Adams.txt",
    "Surgery_Schwartz.txt",
    "InternalMed_Harrison.txt",
]


def collect_textbook_sources():
    sources = []
    for name in TEXTBOOKS:
        path = os.path.join(TEXTBOOK_DIR, name)
        if os.path.exists(path):
            sources.append(("textbook", name, path))
        else:
            print(f"  [MISSING] {path}")
    return sources


def collect_pubmedqa_sources():
    """Read pubmedqa.csv and yield one combined text blob per batch."""
    if not os.path.exists(PUBMEDQA_CSV):
        print(f"  [MISSING] {PUBMEDQA_CSV}")
        return []
    return [("pubmedqa", "pubmedqa.csv", PUBMEDQA_CSV)]


async def ingest_text(rag, label: str, text: str):
    size_mb = len(text.encode()) / 1_048_576
    print(f"  Inserting {size_mb:.1f} MB — {label} ...")
    await rag.ainsert(text)
    print(f"  ✓ Done — {label}")


async def ingest_textbook(rag, path: str, name: str):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    await ingest_text(rag, name, text)


async def ingest_pubmedqa(rag, path: str):
    """Concatenate all abstracts from pubmedqa.csv into a single insert."""
    import csv
    print(f"  Reading {path} ...")
    abstracts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        # Detect abstract column
        fieldnames = reader.fieldnames or []
        abstract_col = next(
            (c for c in fieldnames if "abstract" in c.lower()), None
        )
        if abstract_col is None:
            print(f"  [WARN] No 'abstract' column found in {path}. Columns: {fieldnames}")
            return
        for row in reader:
            val = (row.get(abstract_col) or "").strip()
            if val:
                abstracts.append(val)

    print(f"  Loaded {len(abstracts):,} abstracts from pubmedqa.csv")
    # Insert in batches of 5000 abstracts to avoid a single massive call
    BATCH_SIZE = 5000
    total_batches = (len(abstracts) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, len(abstracts), BATCH_SIZE):
        batch_num = i // BATCH_SIZE + 1
        batch = abstracts[i : i + BATCH_SIZE]
        text = "\n\n".join(batch)
        label = f"pubmedqa batch {batch_num}/{total_batches} ({len(batch)} abstracts)"
        await ingest_text(rag, label, text)


async def main():
    parser = argparse.ArgumentParser(description="Ingest all biomed documents into LightRAG")
    parser.add_argument("--textbooks", action="store_true", help="Ingest textbooks only")
    parser.add_argument("--pubmedqa",  action="store_true", help="Ingest PubMedQA only")
    parser.add_argument("--dry-run",   action="store_true", help="List files without ingesting")
    args = parser.parse_args()

    # Default: both
    do_textbooks = args.textbooks or (not args.textbooks and not args.pubmedqa)
    do_pubmedqa  = args.pubmedqa  or (not args.textbooks and not args.pubmedqa)

    print(f"\n{'='*60}")
    print(f"  LLM model    : {LLM_MODEL}")
    print(f"  Working dir  : {WORKING_DIR}")
    print(f"  max_tokens   : {LLM_MAX_TOKENS}")
    print(f"  Debug log    : {DEBUG_OUTPUT_FILE if DEBUG_LLM else 'disabled'}")
    print(f"  Textbooks    : {'yes' if do_textbooks else 'no'}")
    print(f"  PubMedQA     : {'yes' if do_pubmedqa else 'no'}")
    print(f"{'='*60}\n")

    sources = []
    if do_textbooks:
        sources += collect_textbook_sources()
    if do_pubmedqa:
        sources += collect_pubmedqa_sources()

    if not sources:
        print("No sources found. Exiting.")
        return

    print(f"Sources to ingest ({len(sources)} total):")
    total_size = 0
    for kind, name, path in sources:
        size = os.path.getsize(path)
        total_size += size
        print(f"  [{kind}] {name:45s} {size/1_048_576:6.1f} MB")
    print(f"  {'TOTAL':45s} {total_size/1_048_576:6.1f} MB\n")

    if args.dry_run:
        print("Dry-run mode — no ingestion performed.")
        return

    rag = build_rag()
    await rag.initialize_storages()
    print()

    for idx, (kind, name, path) in enumerate(sources, 1):
        print(f"[{idx}/{len(sources)}] {name}")
        try:
            if kind == "textbook":
                await ingest_textbook(rag, path, name)
            elif kind == "pubmedqa":
                await ingest_pubmedqa(rag, path)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    asyncio.run(main())
