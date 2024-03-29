import time

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import yaml
import gc

from tools import data_loader


def get_features(feature_list, df_yad, df_log, df_candidate, train_test="train", n_pivot=8):
    '''
    df_yado: pl.DataFrame
        yad_no, yado_type, total_room_cnt, wireless_lan_flg, onsen_flg, kd_stn_5min, kd_bch_5min, kd_slp_5min, kd_conv_walk_5min
    df_log: pl.DataFrame
        session_id, seq_no, yad_no
    df_candidate: pl.DataFrame
        session_id, yad_no

    '''

    # yad_features = get_yado_features(yad_f, df_yad)
    # session_features = get_session_features(session_f, df_log)

    enumerated = df_log.with_columns(
        pl.arange(pl.count(), 0, -1).over("session_id").alias("enum")
    )
    pivoted = enumerated.pivot(
        index="session_id", columns="enum", values="yad_no")

    pivot_cols = [f"{i}" for i in range(1, n_pivot+1)]
    pivot_cols.append("session_id")
    pivoted = pivoted.select(pivot_cols)
    df_candidate = df_candidate.join(
        pivoted, how="left", left_on="session_id", right_on="session_id", )
    # df_candidate = df_candidate.rename(
    #     {"1": "yad_no_1", "2": "yad_no_2", "3": "yad_no_3", "4": "yad_no_4", "yad_no": "yad_no_cand"})
    rename_dict = {f"{i}": f"yad_no_{i}" for i in range(1, n_pivot+1)}
    rename_dict["yad_no"] = "yad_no_cand"
    df_candidate = df_candidate.rename(rename_dict)

    # # feature
    for i in range(n_pivot):
        df_candidate = df_candidate.join(
            df_yad, how="left", left_on=f"yad_no_{i+1}", right_on="yad_no", suffix=(f'_{i+1}'))

    df_candidate = df_candidate.rename(
        {f"{col}": f"{col}_1" for col in df_yad.columns if col != "yad_no"})

    df_candidate = df_candidate.join(
        df_yad, how="left", left_on="yad_no_cand", right_on="yad_no", suffix="_cand")

    df_candidate = df_candidate.rename(
        {f"{col}": f"{col}_cand" for col in df_yad.columns if col != "yad_no"})
    #
    # # make matching features which means:
    # # if {yad.columns}_{1,..,4} == {yad.columns}_cand then 1 else 0
    # n_pivot_matching = 2
    # for i in range(n_pivot_matching):
    #     for col in df_yad.columns:
    #         if col != "total_room_cnt" and col != "yad_no":
    #             df_candidate = df_candidate.with_columns(
    #                 (pl.col(f"{col}_{i+1}") == pl.col(f"{col}_cand")
    #                  ).cast(pl.Int8).alias(f"{col}_{i+1}_match")
    #             )

    for i in range(n_pivot):
        df_candidate = df_candidate.with_columns(
            (pl.col(f"yad_no_{i+1}") == pl.col(f"yad_no_cand")
             ).cast(pl.Int8).alias(f"yad_no_{i+1}_match")
        )

    # ## sum matching features
    # matching = (df_log.join(df_candidate, how="left", left_on="session_id", right_on="session_id")
    #     .join(df_yad, how="left", left_on="yad_no_cand", right_on="yad_no")
    #     .join(df_yad, how="left", on="yad_no",))
    # for col in df_yad.columns:
    #     if col != "total_room_cnt" and col!="yad_no":
    #         sum_match = (
    #             matching.filter(pl.col(f"{col}") == pl.col(f"{col}_right"))
    #             .groupby(["session_id", "yad_no_cand"])
    #             .agg(pl.sum(f"{col}").alias(f"{col}_sum_match"))[["session_id", "yad_no_cand", f"{col}_sum_match"]]
    #         )
    #         df_candidate = df_candidate.join(
    #             sum_match, how="left", left_on=["session_id", "yad_no_cand"], right_on=["session_id", "yad_no_cand"])
    #
    # for col in ["yad_no"]:
    #     sum_odd = (
    #         matching
    #         .filter(pl.col(f"{col}") == pl.col(f"{col}_right"))
    #         .filter(pl.col(f"seq_no") % 2 == 1)
    #         .groupby(["session_id", "yad_no_cand"])
    #         .agg(pl.sum(f"{col}").alias(f"{col}_sum_odd"))[["session_id", "yad_no_cand", f"{col}_sum_odd"]]
    #     )
    #     df_candidate = df_candidate.join(
    #         sum_odd, how="left", left_on=["session_id", "yad_no_cand"], right_on=["session_id", "yad_no_cand"])

    # # cat image features
    # image_features = pl.read_parquet(
    #     "data/features/features_image_cluster.parquet")
    #
    # for i in range(1):
    #     df_candidate = df_candidate.join(
    #         image_features, how="left", left_on=f"yad_no_{i+1}", right_on="yad_no")
    #     df_candidate = df_candidate.rename(
    #         {f"{col}": f"{col}_{i+1}" for col in image_features.columns if col != "yad_no"})
    #     df_candidate = df_candidate.with_columns([
    #         pl.col(f"{col}_{i+1}").cast(pl.Int8) for col in image_features.columns if col != "yad_no"])
    #
    # df_candidate = df_candidate.join(
    #     image_features, how="left", left_on="yad_no_cand", right_on="yad_no", suffix="_cand")
    # df_candidate = df_candidate.rename(
    #     {f"{col}": f"{col}_cand" for col in image_features.columns if col != "yad_no"})
    # df_candidate = df_candidate.with_columns([
    #     pl.col(f"{col}_cand").cast(pl.Int8) for col in image_features.columns if col != "yad_no"])
    #
    # # make image category matching features which means:
    # # if both {image_features.columns}_{1,2, 3, 4} and {image_features.columns}_cand are positive then 1 else 0
    # for i in range(1):
    #                 # 0 to 9
    #     for col in ["image_cluster_0", "image_cluster_1", "image_cluster_2", "image_cluster_3", "image_cluster_4", "image_cluster_5", "image_cluster_6", "image_cluster_7", "image_cluster_8", "image_cluster_9"]:
    #         if col != "yad_no":
    #             df_candidate = df_candidate.with_columns(
    #                 (pl.col(f"{col}_{i+1}") * pl.col(f"{col}_cand")
    #                  ).cast(pl.Int8).alias(f"{col}_{i+1}_match")
    #             )
    #
    # # delete raw image feature
    # for i in range(1):
    #     for col in image_features.columns:
    #         if col != "yad_no":
    #             df_candidate = df_candidate.drop(f"{col}_{i+1}")
    #             df_candidate = df_candidate.drop(f"{col}_cand")
    #
    # ## area embedding matching features

    # for area in ["lrg_cd", "sml_cd"]:
    #     cd_emb = pl.read_parquet(
    #         f"data/item2vec/area_emb_{area}.parquet")
    #
    #     df_candidate = df_candidate.join(
    #         cd_emb, how="left", left_on=f"{area}_cand", right_on="area")
    #     df_candidate = df_candidate.rename(
    #         {f"{col}": f"{col}_{area}_cand" for col in cd_emb.columns if col != "area"})
    #     df_candidate = df_candidate.with_columns([
    #         pl.col(f"{col}_{area}_cand").cast(pl.Float32) for col in cd_emb.columns if col != "area"])
    #
    #     for i in range(2):
    #         df_candidate = df_candidate.join(
    #             cd_emb, how="left", left_on=f"{area}_{i+1}", right_on={"area"})
    #         df_candidate = df_candidate.rename(
    #             {f"{col}": f"{col}_{area}_{i+1}" for col in cd_emb.columns if col != "area"})
    #         df_candidate = df_candidate.with_columns([
    #             pl.col(f"{col}_{area}_{i+1}").cast(pl.Float32) for col in cd_emb.columns if col != "area"])
    #         # cosin similarity
    #         # col_prod = [sum([pl.col(f"area_emb_{j+1}_{area}_{i+1}") * pl.col(f"area_emb_{j+1}_{area}_cand") for j in range(5)]).cast(pl.Float32).alias(f"area_emb_inner_{area}_{i+1}")]
    #         col_sim = [(sum([pl.col(f"area_emb_{j+1}_{area}_{i+1}") * pl.col(f"area_emb_{j+1}_{area}_cand") for j in range(5)]) / sum([pl.col(f"area_emb_{j+1}_{area}_{i+1}")
    #                                                                                                                                    ** 2 for j in range(5)]) ** 0.5 / sum([pl.col(f"area_emb_{j+1}_{area}_{i+1}") ** 2 for j in range(5)]) ** 0.5).cast(pl.Float32).alias(f"{area}_emb_sim_{i+1}")]
    #         df_candidate = df_candidate.with_columns(
    #             col_sim
    #         )
    #
    #         # # check added column
    #         # for col in df_candidate.columns:
    #         #     if col.startswith(f"{area}_emb_sim_{i}"):
    #         #         print(df_candidate.select(col))
    #
    #     #   drop
    #     df_candidate = df_candidate.drop(
    #         [f"area_emb_{j+1}_{area}_{i+1}" for j in range(5) for i in range(n_pivot)])
    #     df_candidate = df_candidate.drop(
    #         [f"area_emb_{j+1}_{area}_cand" for j in range(5)])
    # #
    #
    # # sum matching features
    # for col in image_features.columns:
    #     if col != "yad_no":
    #         sum_col = sum([pl.col(f"{col}_{i+1}_match")
    #                        for i in range(n_pivot)])
    #         df_candidate = df_candidate.with_columns(
    #             sum_col.cast(pl.Int8).alias(f"{col}_match_sum")
    #         )
    # # drop raw image features
    # df_candidate = df_candidate.drop(
    #     [f"{col}_{i+1}" for col in image_features.columns for i in range(n_pivot) if col != "yad_no"])
    # df_candidate = df_candidate.drop(
    #     [f"{col}_cand" for col in image_features.columns if col != "yad_no"])
    #
    # session_length_each_yad

    # length_each_yad = pl.read_parquet(
    #     "data/features/session_length_each_yad.parquet")
    #
    # for i in range(2):
    #     df_candidate = df_candidate.join(
    #         length_each_yad, how="left", left_on=f"yad_no_{i+1}", right_on="yad_no")
    #     df_candidate = df_candidate.rename(
    #         {f"{col}": f"{col}_{i+1}" for col in length_each_yad.columns if col != "yad_no"})
    #
    # df_candidate = df_candidate.join(
    #     length_each_yad, how="left", left_on="yad_no_cand", right_on="yad_no")
    # df_candidate = df_candidate.rename(
    #     {f"{col}": f"{col}_cand" for col in length_each_yad.columns if col != "yad_no"})
    #
    # covisitation feature

    # n_covisit_pivot = 3
    #
    # for i in range(n_covisit_pivot):
    #     covisit = pl.read_parquet(
    #         f"data/features/{train_test}_covisit_features.parquet")
    #     # covisit = pl.read_parquet(
    #     #     f"data/features/test_covisit_features.parquet")
    #     df_candidate = (df_candidate.join(
    #         covisit, how="left", left_on=[f"yad_no_{i+1}", "yad_no_cand"], right_on=["latest_yad_no", "yad_no"]).with_columns(
    #         pl.col("co_visit_count").cast(pl.Int16)).rename({"co_visit_count": f"co_visit_count_{i+1}_cand"})
    #     )
    #
    # # next seq feature
    # next_seq = pl.read_parquet(
    #     f"data/features/{train_test}_next_seq_features.parquet")
    # next_seq = pl.read_parquet(
    #     f"data/features/test_next_seq_features.parquet")
    # ## convert to rank
    # # next_seq = next_seq.with_columns(
    # #     (pl.col("count").rank()*128/next_seq.shape[0]).cast(pl.UInt8)
    # # )
    # df_candidate = (df_candidate.join(
    #     next_seq, how="left", left_on=["yad_no_1", "yad_no_cand"], right_on=["yad_no", "next_yad_no"]).with_columns(
    #     pl.col("count").cast(pl.Int16)).rename({"count": "next_seq_count_1_cand"})
    # )
    #
    # make feature in discussion

    # sequence number of each session
    max_seq = df_log.group_by("session_id").agg(
        pl.max("seq_no").alias("max_seq"))[["session_id", "max_seq"]].with_columns(pl.col("max_seq").cast(pl.Int8))

    df_candidate = df_candidate.join(
        max_seq, how="left", left_on="session_id", right_on="session_id")

    # first and last appearance of candidate in log
    first_cand_appear_seq_in_log = df_candidate.join(
        df_log, how="inner", left_on=["session_id", "yad_no_cand"], right_on=["session_id", "yad_no"]).group_by(["session_id", "yad_no_cand"]).agg(pl.max("seq_no").alias("first_cand_appear_seq_in_log"))[["session_id", "yad_no_cand", "first_cand_appear_seq_in_log"]]

    last_cand_appear_seq_in_log = df_candidate.join(
        df_log, how="inner", left_on=["session_id", "yad_no_cand"], right_on=["session_id", "yad_no"]).group_by(["session_id", "yad_no_cand"]).agg(pl.min("seq_no").alias("last_cand_appear_seq_in_log"))[["session_id", "yad_no_cand", "last_cand_appear_seq_in_log"]]


    df_candidate = df_candidate.join(
        first_cand_appear_seq_in_log, how="left", on=["session_id", "yad_no_cand"]).with_columns(
        pl.col("first_cand_appear_seq_in_log").cast(pl.Int8))

    df_candidate = df_candidate.join(
        last_cand_appear_seq_in_log, how="left", on=["session_id", "yad_no_cand"]).with_columns(
        pl.col("last_cand_appear_seq_in_log").cast(pl.Int8))


    df_candidate = df_candidate.with_columns(
        (pl.col("max_seq") - pl.col("first_cand_appear_seq_in_log")).alias("seq_no_diff"))

    del first_cand_appear_seq_in_log, last_cand_appear_seq_in_log
    gc.collect()

    # view number of candidate in log
    view_num_of_cand_in_log = df_candidate[["session_id", "yad_no_cand"]].join(
        df_log, how="inner", left_on=["session_id", "yad_no_cand"], right_on=["session_id", "yad_no"]).group_by(["session_id", "yad_no_cand"]).count().rename({"count": "view_num_of_cand_in_log"})

    view_num_of_cand_in_log = view_num_of_cand_in_log.with_columns(
        (pl.col("view_num_of_cand_in_log").rank() * 127
            / view_num_of_cand_in_log.shape[0]).cast(pl.UInt8, strict=False)
    )

    df_candidate = df_candidate.join(
        view_num_of_cand_in_log, how="left", on=["session_id", "yad_no_cand"]).with_columns(
        pl.col("view_num_of_cand_in_log").cast(pl.UInt8))

    # make feature in discussion
    feature_name_list = [
        'latest_next_booking_top20',
        'top20_popular_yado',
        'top10_wid_popular_yado',
        'top10_ken_popular_yado',
        'top10_lrg_popular_yado',
        'top10_sml_popular_yado'
    ]

    for feature_name in tqdm(feature_name_list):
        print(feature_name)
        feature = pl.read_parquet(
            f'data/features/{train_test}_{feature_name}_feature.parquet')

        print(feature.to_pandas().describe())

        # TODO: nullの処理
        if train_test == 'train':
            # for fold in range(CFG.fold_num):
            if 'session_id' in feature.columns:
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                    'session_id', 'yad_no_cand'], right_on=["session_id", "yad_no"])

            elif 'latest_yad_no' in feature.columns:
                feature = feature.with_columns(pl.col("fold").cast(pl.Int64), )
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                    'fold', 'yad_no_1', 'yad_no_cand'], right_on=['fold', 'latest_yad_no', 'yad_no'])
            else:
                feature = feature.with_columns(pl.col("fold").cast(pl.Int64), )
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                    'fold', 'yad_no_cand'], right_on=['fold', 'yad_no'])
        else:
            if 'session_id' in feature.columns:
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                    'session_id', 'yad_no_cand'], right_on=["session_id", "yad_no"])
            elif 'latest_yad_no' in feature.columns:
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                    'yad_no_1', 'yad_no_cand'], right_on=['latest_yad_no', 'yad_no'])
            else:
                df_candidate = df_candidate.join(feature, how='left', left_on=[
                    'yad_no_cand'], right_on=['yad_no'])
    df_candidate = df_candidate.drop(["wid_cd", "ken_cd", "lrg_cd", "sml_cd"])

    df_candidate = df_candidate.with_columns([
        pl.col("latest_next_booking_rank").cast(pl.Int8),
        pl.col("popular_rank").cast(pl.Int16),
        pl.col("popular_wid_cd_rank").cast(pl.Int16),
        pl.col("popular_ken_cd_rank").cast(pl.Int16),
        pl.col("popular_lrg_cd_rank").cast(pl.Int8),
        pl.col("popular_sml_cd_rank").cast(pl.Int8),
    ])

    df_candidate = df_candidate.with_columns([
        pl.col(f"total_room_cnt_{i+1}").cast(pl.Int16) for i in range(n_pivot) if f"total_room_cnt_{i+1}" in df_candidate.columns]).with_columns(pl.col(f"total_room_cnt_cand").cast(pl.Int16))

    # drop many category columns
    cand_dont_drop_cols = [
        'yad_type', 'total_room_cnt', 'wireless_lan_flg', 'onsen_flg', 'kd_stn_5min', 'kd_bch_5min', 'kd_slp_5min', 'kd_conv_walk_5min', 'wid_cd', 'ken_cd', 'lrg_cd']

    pivot_dont_drop_cols = [
        "wid_cd_1", "ken_cd_1", "lrg_cd_1",
    ]

    for i in range(n_pivot):
        for col in df_yad.columns:
            if f"{col}_{i+1}" not in pivot_dont_drop_cols and f"{col}_{i+1}" in df_candidate.columns:
                df_candidate = df_candidate.drop(f"{col}_{i+1}")

    for col in df_yad.columns:
        if col not in cand_dont_drop_cols and col != "yad_no" and f"{col}_cand" in df_candidate.columns:
            df_candidate = df_candidate.drop(f"{col}_cand")

    # df_candidate=df_candidate.drop(
    #     ["wid_cd", "ken_cd", "lrg_cd", "sml_cd", ])
    # change to category
    df_candidate = df_candidate.with_columns([
        pl.col("wid_cd_cand").cast(pl.Categorical),
        pl.col("ken_cd_cand").cast(pl.Categorical),
        pl.col("lrg_cd_cand").cast(pl.Categorical),
        pl.col("wid_cd_1").cast(pl.Categorical),
        pl.col("ken_cd_1").cast(pl.Categorical),
        pl.col("lrg_cd_1").cast(pl.Categorical),
        # pl.col("sml_cd").cast(pl.Categorical),
        pl.col("wireless_lan_flg_cand").cast(pl.Int8),
        pl.col("onsen_flg_cand").cast(pl.Int8),
        pl.col("kd_stn_5min_cand").cast(pl.Int8),
        pl.col("kd_bch_5min_cand").cast(pl.Int8),
        pl.col("kd_slp_5min_cand").cast(pl.Int8),
        pl.col("kd_conv_walk_5min_cand").cast(pl.Int8),
    ])

    if train_test == "train":
        df_candidate = df_candidate.drop("fold")

    return df_candidate


def make_feature_and_label(candidate_train_tstamp, candidate_test_tstamp, featuer_list=None):

    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    label_train = loader.load_cv_label()
    yado = loader.load_yado()
    test_log = loader.load_test_log()

    candidate = pl.read_parquet(
        f'data/candidates/train_candidate_{candidate_train_tstamp}.parquet')
    candidate_test = pl.read_parquet(
        f'data/candidates/test_candidate_{candidate_test_tstamp}.parquet')

    features_train = get_features(featuer_list, yado, train_log, candidate.join(
        label_train.select(["session_id", "fold"]), on=["session_id"]), train_test="train")
    features_test = get_features(
        featuer_list, yado, test_log, candidate_test, train_test="test")

    label_train = label_train.rename(
        {"yad_no": "yad_no_ans", })
    # if candidate["session_id", "yad_no"] is in train_label: then 1 else 0

    label_train = (
        candidate
        .select("session_id", "yad_no")
        .rename({"yad_no": "yad_no_cand"})
        .join(label_train, how="left", on="session_id")
        .select("session_id", "yad_no_cand", "yad_no_ans", "fold")
        .with_columns(
            pl.when(pl.col("yad_no_cand") == pl.col("yad_no_ans")).then(
                1).otherwise(0).alias("label")
        )
        .select(["session_id", "yad_no_cand", "fold", "label"])
        .sort("session_id", "yad_no_cand")
    )

    null_columns = set([
        col for col in features_train.columns if features_train[col].null_count() > 0])
    null_columns_test = set([
        col for col in features_test.columns if features_test[col].null_count() > 0])

    print("null features", null_columns)
    print("null features test", null_columns_test)
    print(null_columns_test - null_columns)
    print(features_train.columns)

    features_train = features_train.lazy().sort(
        "session_id", "yad_no_cand").collect()
    features_test = features_test.lazy().sort(
        "session_id", "yad_no_cand").collect()

    timestamp = time.strftime("%Y%m%d%H%M%S")
    features_train.write_parquet(
        f'data/features/X_train_{candidate_train_tstamp}_{timestamp}.parquet')
    features_test.write_parquet(
        f'data/features/X_test_{candidate_test_tstamp}_{timestamp}.parquet')
    label_train.write_parquet(
        f'data/features/y_train_{candidate_train_tstamp}_{timestamp}.parquet')

    with open("data/features/latest.yaml", "w") as f:
        yaml.dump({
            "X_train": f"data/features/X_train_{candidate_train_tstamp}_{timestamp}.parquet",
            "X_test": f"data/features/X_test_{candidate_test_tstamp}_{timestamp}.parquet",
            "y_train": f"data/features/y_train_{candidate_train_tstamp}_{timestamp}.parquet",
        }, f)


if __name__ == "__main__":
    make_feature_and_label("rule_based", "rule_based")
