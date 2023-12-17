import pandas as pd
from collections import defaultdict
from heapq import heappush, heappop

from tools.data_loader import AtmaData16Loader
import polars as pl


def create_top_10_yad_predict(_df):

    print(_df)
    _agg = _df.groupby("session_id")["yad_no"].apply(list)

    out_df = pd.DataFrame(
        index=_agg.index, data=_agg.values.tolist())

    return out_df


if __name__ == "__main__":

    # yado = pd.read_csv("in/yado.csv", dtype={"yad_no":int}) # 今回は使いません

    # test_session = pl.read_csv("data/test_session.csv", dtype={"session_id":str})
    # yado = pl.read_csv("data/yado.csv", dtype={"yad_no":int}) # 今回は使いません
    # train_log = pl.read_csv("data/train_log.csv", dtype={"session_id":str, "seq_no":int, "yad_no":int})
    # train_label = pl.read_csv("data/train_label.csv", dtype={"session_id":str, "yad_no":int})
    # test_log = pl.read_csv("data/test_log.csv", dtype={"session_id":str, "seq_no":int, "yad_no":int})
    test_session = pl.read_csv("data/test_session.csv",)

    loader = AtmaData16Loader()
    train_log = loader.load_train_log()
    label_train = loader.load_cv_label()
    yado = loader.load_yado()
    test_log = loader.load_test_log()

    train_log_len = (
        train_log
        .group_by("session_id")
        .count()
        .select(["session_id", "count"])
        .rename({"count": "seq_count"},)
    )
    log_last_to_label = (
        train_log
        .join(train_log_len, on="session_id", how="left")
        .filter(pl.col("seq_no") == pl.col("seq_count") - 1)
        .join(label_train, on="session_id", how="left")
        .select(["yad_no", "yad_no_right"])
        .rename({"yad_no": "yad_no_latest"},)
        .rename({"yad_no_right": "yad_no"},)
    )

    log_last_to_label_count = (
        log_last_to_label
        .group_by(["yad_no_latest", "yad_no"])
        .count()
        .sort(by="count",)
    )
    print("log_last_to_label_count", log_last_to_label_count)

    test_log_len = (
        test_log
        .group_by("session_id")
        .count()
        .select(["session_id", "count"])
        .rename({"count": "seq_count"},)
    )

    # take most large seq_no row for each session
    test_log_last = (
        test_log
        .join(test_log_len, on="session_id", how="left")
        .filter(pl.col("seq_no") == pl.col("seq_count") - 1)
        .select(["session_id", "yad_no"])
    )

    test_log_last_to_label = (
        test_log_last
        .join(log_last_to_label_count, left_on="yad_no", right_on="yad_no_latest", how="inner")
        .select(["session_id", "yad_no_right", "count"])
        .rename({"yad_no_right": "yad_no"},)
        .with_columns(pl.lit(7).alias("priority"))
    )

    print("test_log_last_to_label", test_log_last_to_label.sort(
        by=["session_id", "count"], descending=[False, True]))

    test_log_second_last = (
        test_log
        .join(test_log_len, on="session_id", how="left")
        .filter(pl.col("seq_no") == pl.col("seq_count") - 2)
        .select(["session_id", "yad_no"])
        .with_columns(pl.lit(0, dtype=pl.UInt32).alias("count"))
        .with_columns(pl.lit(10).alias("priority"))
    )

    # test_log_odd_last = (
    #     test_log
    #     .join(test_log_len, on="session_id", how="left")
    #     .filter((pl.col("seq_count") - pl.col("seq_no")) % 2 == 0)
    #     .select(["session_id", "yad_no"])
    #     .with_columns(pl.lit(0, dtype=pl.UInt32).alias("count"))
    #     .with_columns(pl.lit(9).alias("priority"))
    # )

    popular_in_sml_cd = (
        label_train
        .join(yado, on="yad_no", how="left")
        .select(["sml_cd", "yad_no"])
        .group_by(["sml_cd", "yad_no"])
        .count()
        .sort(by="count", descending=True)
        .select(["sml_cd", "yad_no", "count"])
    )
    popular_in_lrg_cd = (
        label_train
        .join(yado, on="yad_no", how="left")
        .select(["lrg_cd", "yad_no"])
        .group_by(["lrg_cd", "yad_no"])
        .count()
        .sort(by="count", descending=True)
        .select(["lrg_cd", "yad_no", "count"])
    )

    popular_in_wid_cd = (
        label_train
        .join(yado, on="yad_no", how="left")
        .select(["wid_cd", "yad_no"])
        .group_by(["wid_cd", "yad_no"])
        .count()
        .sort(by="count", descending=True)
        .group_by("wid_cd")
        .head(10)
        .select(["wid_cd", "yad_no", "count"])
    )

    popular_in_sml_cd_cand = (
        test_log
        .join(yado, on="yad_no", how="left")
        .select(["session_id", "sml_cd"])
        .join(popular_in_sml_cd, on="sml_cd", how="left")
        .select(["session_id", "yad_no", "count"])
        .unique(["session_id", "yad_no"])
        .with_columns(pl.lit(4).alias("priority"))
    )

    popular_in_lrg_cd_cand = (
        test_log
        .join(yado, on="yad_no", how="left")
        .select(["session_id", "lrg_cd"])
        .join(popular_in_lrg_cd, on="lrg_cd", how="left")
        .select(["session_id", "yad_no", "count"])
        .unique(["session_id", "yad_no"])
        .with_columns(pl.lit(3).alias("priority"))
    )

    popular_in_wid_cd_cand = (
        test_log
        .join(yado, on="yad_no", how="left")
        .select(["session_id", "wid_cd"])
        .join(popular_in_wid_cd, on="wid_cd", how="left")
        .select(["session_id", "yad_no", "count"])
        .unique(["session_id", "yad_no"])
        .with_columns(pl.lit(2).alias("priority"))
    )

    # network
    network_cand_top_20 = pl.read_parquet(
        "data/candidates/train_network_top20_candidates.parquet")
    print(network_cand_top_20)

    network_cand_top_20 = (
        test_log.join(network_cand_top_20, on="yad_no", how="inner")
        .select(["session_id", "candidate_yad_no"])
        .rename({"candidate_yad_no": "yad_no"})
        .with_columns(pl.lit(0, dtype=pl.UInt32).alias("count"))
        .with_columns(pl.lit(5).alias("priority"))
    )

    next_seq = pl.read_parquet(
        "data/candidates/test_next_seq_top10_candidates.parquet")

    print(next_seq)
    next_seq = (
        test_log.join(next_seq, left_on="yad_no", right_on="latest_yad_no", how="inner")
        .join(test_log_len, on="session_id", how="left")
        .filter(pl.col("seq_no") == pl.col("seq_count") - 1)
        .select(["session_id", "yad_no"])
        .with_columns(pl.lit(0, dtype=pl.UInt32).alias("count"))
        .with_columns(pl.lit(6).alias("priority"))
    )

    # concat all
    # print(test_log_last_to_label.to_pandas().isnull().sum())
    # print(test_log_second_last.to_pandas().isnull().sum())
    # print(popular_in_sml_cd_cand.to_pandas().isnull().sum())
    # print(popular_in_lrg_cd_cand.to_pandas().isnull().sum())
    # print(popular_in_wid_cd_cand.to_pandas().isnull().sum())

    # if same session_id and yad_no, then take max priority
    candidates = pl.concat([
        test_log_last_to_label,
        test_log_second_last,
        popular_in_sml_cd_cand,
        popular_in_lrg_cd_cand,
        popular_in_wid_cd_cand,
        network_cand_top_20
    ])

    max_priority = (
        candidates
        .group_by(["session_id", "yad_no"])
        .agg(pl.max("priority").alias("priority"))
        .select(["session_id", "yad_no", "priority"])
    )

    candidates = (
        candidates
        .join(max_priority, on=["session_id", "yad_no", "priority"], how="inner")
        .select(["session_id", "yad_no", "priority", "count"])
    )

    # sort by priority and count, then take top 10 for each session
    candidates = (
        candidates
        .sort(by=["session_id", "priority", "count"], descending=[False, True, True])
        .group_by("session_id", maintain_order=True)
        .head(10)
        .sort(by=["session_id", "priority", "count"], descending=[False, True, True])
        .select(["session_id", "yad_no"])
    )

    top10 = create_top_10_yad_predict(candidates.to_pandas())
    top10.columns = [f'predict_{c}' for c in top10.columns]
    print(top10)

    # is there null in top10?
    print(top10.isnull().sum())
    # print null rows
    print(top10[top10.isnull().any(axis=1)])

    top10.to_csv("rule_candidate.csv", index=False)
