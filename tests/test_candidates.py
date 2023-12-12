import polars as pl


print(pl.read_parquet("data/candidates/train_candidate_20231211.parquet").group_by("session_id").count())
