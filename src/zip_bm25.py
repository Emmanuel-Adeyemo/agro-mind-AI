import gzip
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BM25_PATH = ROOT / 'bm25_index.pkl'
OUT_PATH = ROOT / 'bm25_index.pkl.gz'

if BM25_PATH.exists():

    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    with gzip.open(OUT_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"Compression Complete! New size: {OUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")
else:
    print("Could not find 'bm25_index.pkl' in the current working directory.")