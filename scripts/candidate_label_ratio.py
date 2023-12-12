import polars as pl
import argparse

from tools import data_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_train_tstamp", type=str, default="20231211")
    parser.add_argument("--candidate_test_tstamp", type=str, default="20231211")
    args = parser.parse_args()

    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    label_train = loader.load_cv_label()
    yado = loader.load_yado()
    test_log = loader.load_test_log()

    candidate = pl.read_parquet(
        f'data/candidates/train_candidate_{args.candidate_train_tstamp}.parquet')

    candidate = candidate.join(label_train, on="session_id").sort(by="session_id")

    # add column match, which is 1 if yad_no == yad_no_right
    candidate = candidate.with_columns(
        pl.when(pl.col("yad_no") == pl.col("yad_no_right")).then(1).otherwise(0).alias("match")
    ).group_by("session_id").agg(pl.sum("match").alias("match_sum")).with_columns(
        pl.when(pl.col("match_sum") > 0).then(1).otherwise(0).alias("cand_session")
    )

    # take average of cand_session
    print(candidate.select("cand_session").mean())




