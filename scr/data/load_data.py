# src/data/load_data.py

import os
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, base_dir="data/raw"):
        self.base_dir = base_dir

    def _full_path(self, filename):
        path = os.path.join(self.base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path

    def load_npz(self, filename):
        data = np.load(self._full_path(filename), allow_pickle=True)
        print(f"[INFO] Loaded .npz file: {filename}, keys: {list(data.keys())}")
        return data

    def load_parquet(self, filename):
        df = pd.read_parquet(self._full_path(filename))
        print(f"[INFO] Loaded Parquet file: {filename}, shape: {df.shape}")
        return df

    def load_gff3(self, filename):
        df = pd.read_csv(
            self._full_path(filename),
            sep="\t",
            comment="#",
            header=None,
            names=[
                "seqid", "source", "type", "start", "end",
                "score", "strand", "phase", "attributes"
            ]
        )
        print(f"[INFO] Loaded GFF3 file: {filename}, shape: {df.shape}")
        return df
