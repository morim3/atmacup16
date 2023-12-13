from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import KFold
from tqdm import tqdm

input_dir = Path("data")

DataFrame: TypeAlias = pl.DataFrame | pd.DataFrame
DfType: TypeAlias = Literal["pl", "pd"]


class AtmaData16Loader:
    def __init__(self, ):
        self.input_dir = input_dir
        self.csv_paths = input_dir.glob("*.csv")

    def csv2parquet(self):
        for csv_path in tqdm(self.csv_paths):
            parquet_path = self.input_dir / Path(csv_path.stem + ".parquet")
            pl.read_csv(csv_path).write_parquet(parquet_path)

    def save_cv_label(self, frame_type: DfType = "pl") -> DataFrame:
        train_label = self.load_train_label(frame_type)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        fold_assign = np.ones(train_label.shape[0], dtype="int")
        for i, (train_idx, valid_idx) in enumerate(kf.split(train_label, )):
            fold_assign[valid_idx] = i

        train_label = train_label.with_columns(
            pl.Series("fold", fold_assign),
        )

        train_label.write_parquet(self.input_dir / "train_label_cv.parquet")

    def load_cv_label(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_parquet(self.input_dir / "train_label_cv.parquet", frame_type)

    @staticmethod
    def _load_parquet(path: Path, frame_type: DfType) -> DataFrame:
        return pl.read_parquet(path) if frame_type == "pl" else pd.read_parquet(path)

    def load_test_log(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_parquet(self.input_dir / "test_log.parquet", frame_type)

    def load_train_log(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_parquet(self.input_dir / "train_log.parquet", frame_type)

    def load_all_log(self, frame_type: DfType = "pl") -> DataFrame:
        df = pl.concat([self.load_train_log("pl"), self.load_test_log("pl")])
        return df if frame_type == "pl" else df.to_pandas()

    def load_ses2idx(self) -> tuple[dict[int, str], dict[str, int]]:
        idx2ses = dict(enumerate(self.load_all_log("pd")["session_id"].unique()))
        ses2idx = {k: idx for idx, k in idx2ses.items()}
        assert ses2idx["000007603d533d30453cc45d0f3d119f"] == 0
        assert idx2ses[0] == "000007603d533d30453cc45d0f3d119f"
        return idx2ses, ses2idx

    def load_train_label(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_parquet(self.input_dir / "train_label.parquet", frame_type)

    def load_yado(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_parquet(self.input_dir / "yado.parquet", frame_type)

    def load_all_dfs(self, frame_type: DfType = "pl") -> dict[str, DataFrame]:
        return {path.stem: self._load_parquet(path, frame_type) for path in self.input_dir.glob("*.parquet")}

    def load_sample_submission(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_parquet(self.input_dir / "sample_submission.parquet", frame_type)

    def load_image_features(self, frame_type: DfType = "pl") -> DataFrame:
        return self._load_parquet(self.input_dir / "image_embeddings.parquet", frame_type)



