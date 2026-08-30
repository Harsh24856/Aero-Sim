import torch
import pyarrow.parquet as pq
import os

print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

DATA_DIR = "/Users/harsh/Documents/UAV_Engine/validation/chunks"
SCALER_PATH = os.path.join(DATA_DIR, "scaler.pkl")

files = ["train_chunk_01.parquet", "train_chunk_02.parquet", "train_chunk_03.parquet",
         "train_chunk_04.parquet", "val.parquet", "test.parquet"]

for fname in files:
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        pf = pq.ParquetFile(path)
        size_gb = os.path.getsize(path) / 1e9
        print(fname + ": rows=" + str(pf.metadata.num_rows) + " row_groups=" + str(pf.num_row_groups) + " size=" + str(round(size_gb,2)) + "GB")
    else:
        print(fname + ": NOT FOUND at " + path)

print("scaler.pkl found:", os.path.exists(SCALER_PATH))
print("scaler.pkl path checked:", SCALER_PATH)
