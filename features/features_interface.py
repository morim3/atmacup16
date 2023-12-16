import time

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
import yaml

from tools import data_loader


def get_features(feature_list, df_yad, df_log, df_candidate, train_test="train", n_pivot=5):
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

    # feature
    for i in range(n_pivot):
        df_candidate = df_candidate.join(
            df_yad, how="left", left_on=f"yad_no_{i+1}", right_on="yad_no", suffix=(f'_{i+1}'))

    df_candidate = df_candidate.rename(
        {f"{col}": f"{col}_1" for col in df_yad.columns if col != "yad_no"})

    df_candidate = df_candidate.join(
        df_yad, how="left", left_on="yad_no_cand", right_on="yad_no", suffix="_cand")

    df_candidate = df_candidate.rename(
        {f"{col}": f"{col}_cand" for col in df_yad.columns if col != "yad_no"})

    # make matching features which means:
    # if {yad.columns}_{1,..,4} == {yad.columns}_cand then 1 else 0
    for i in range(n_pivot):
        for col in df_yad.columns:
            df_candidate = df_candidate.with_columns(
                (pl.col(f"{col}_{i+1}") == pl.col(f"{col}_cand")
                 ).cast(pl.Int8).alias(f"{col}_{i+1}_match")
            )

    # ## sum matching features
    for col in df_yad.columns:
        sum_col = sum([pl.col(f"{col}_{i+1}_match")
                      for i in range(0, n_pivot)])
        df_candidate = df_candidate.with_columns(
            sum_col.cast(pl.Int8).alias(f"{col}_match_sum")
        )

    for col in ["yad_no"]:
        sum_col = sum([pl.col(f"{col}_{i+1}_match")
                      for i in range(1, n_pivot, 2)])
        df_candidate = df_candidate.with_columns(
            sum_col.cast(pl.Int8).alias(f"{col}_match_sum_odd")
        )

    # cat image features
    image_features = pl.read_parquet(
        "data/features/features_image_cluster.parquet")

    for i in range(n_pivot):
        df_candidate = df_candidate.join(
            image_features, how="left", left_on=f"yad_no_{i+1}", right_on="yad_no")
        df_candidate = df_candidate.rename(
            {f"{col}": f"{col}_{i+1}" for col in image_features.columns if col != "yad_no"})
        df_candidate = df_candidate.with_columns([
            pl.col(f"{col}_{i+1}").cast(pl.Int8) for col in image_features.columns if col != "yad_no"])

    df_candidate = df_candidate.join(
        image_features, how="left", left_on="yad_no_cand", right_on="yad_no", suffix="_cand")
    df_candidate = df_candidate.rename(
        {f"{col}": f"{col}_cand" for col in image_features.columns if col != "yad_no"})
    df_candidate = df_candidate.with_columns([
        pl.col(f"{col}_cand").cast(pl.Int8) for col in image_features.columns if col != "yad_no"])

    # make image category matching features which means:
    # if both {image_features.columns}_{1,2, 3, 4} and {image_features.columns}_cand are positive then 1 else 0
    # for i in range(1):
    #     for col in image_features.columns:
    #         if col != "yad_no":
    #             df_candidate = df_candidate.with_columns(
    #                 (pl.col(f"{col}_{i+1}") * pl.col(f"{col}_cand")
    #                  ).cast(pl.Int8).alias(f"{col}_{i+1}_match")
    #             )

    # ## area embedding matching features

    for area in ["lrg_cd", "sml_cd"]:
        cd_emb = pl.read_parquet(
            f"data/item2vec/area_emb_{area}.parquet")

        df_candidate = df_candidate.join(
            cd_emb, how="left", left_on=f"{area}_cand", right_on="area")
        df_candidate = df_candidate.rename(
            {f"{col}": f"{col}_{area}_cand" for col in cd_emb.columns if col != "area"})
        df_candidate = df_candidate.with_columns([
            pl.col(f"{col}_{area}_cand").cast(pl.Float32) for col in cd_emb.columns if col != "area"])

        for i in range(n_pivot):
            df_candidate = df_candidate.join(
                cd_emb, how="left", left_on=f"{area}_{i+1}", right_on={"area"})
            df_candidate = df_candidate.rename(
                {f"{col}": f"{col}_{area}_{i+1}" for col in cd_emb.columns if col != "area"})
            df_candidate = df_candidate.with_columns([
                pl.col(f"{col}_{area}_{i+1}").cast(pl.Float32) for col in cd_emb.columns if col != "area"])
            # cosin similarity
            # col_prod = [sum([pl.col(f"area_emb_{j+1}_{area}_{i+1}") * pl.col(f"area_emb_{j+1}_{area}_cand") for j in range(5)]).cast(pl.Float32).alias(f"area_emb_inner_{area}_{i+1}")]
            col_sim = [(sum([pl.col(f"area_emb_{j+1}_{area}_{i+1}") * pl.col(f"area_emb_{j+1}_{area}_cand") for j in range(5)]) / sum([pl.col(f"area_emb_{j+1}_{area}_{i+1}")
                                                                                                                                       ** 2 for j in range(5)]) ** 0.5 / sum([pl.col(f"area_emb_{j+1}_{area}_{i+1}") ** 2 for j in range(5)]) ** 0.5).cast(pl.Float32).alias(f"{area}_emb_sim_{i+1}")]
            df_candidate = df_candidate.with_columns(
                col_sim
            )

            # # check added column
            # for col in df_candidate.columns:
            #     if col.startswith(f"{area}_emb_sim_{i}"):
            #         print(df_candidate.select(col))

        #   drop
        df_candidate = df_candidate.drop(
            [f"area_emb_{j+1}_{area}_{i+1}" for j in range(5) for i in range(n_pivot)])
        df_candidate = df_candidate.drop(
            [f"area_emb_{j+1}_{area}_cand" for j in range(5)])

    #
    # # sum matching features
    # for col in image_features.columns:
    #     if col != "yad_no":
    #         sum_col = sum([pl.col(f"{col}_{i+1}_match")
    #                        for i in range(n_pivot)])
    #         df_candidate = df_candidate.with_columns(
    #             sum_col.cast(pl.Int8).alias(f"{col}_match_sum")
    #         )
    # drop raw image features
    df_candidate = df_candidate.drop(
        [f"{col}_{i+1}" for col in image_features.columns for i in range(n_pivot) if col != "yad_no"])
    df_candidate = df_candidate.drop(
        [f"{col}_cand" for col in image_features.columns if col != "yad_no"])

    # session_length_each_yad

    length_each_yad = pl.read_parquet(
        "data/features/session_length_each_yad.parquet")

    for i in range(n_pivot):
        df_candidate = df_candidate.join(
            length_each_yad, how="left", left_on=f"yad_no_{i+1}", right_on="yad_no")
        df_candidate = df_candidate.rename(
            {f"{col}": f"{col}_{i+1}" for col in length_each_yad.columns if col != "yad_no"})

    df_candidate = df_candidate.join(
        length_each_yad, how="left", left_on="yad_no_cand", right_on="yad_no")
    df_candidate = df_candidate.rename(
        {f"{col}": f"{col}_cand" for col in length_each_yad.columns if col != "yad_no"})

    # covisitation feature

    n_covisit_pivot = 3

    for i in range(n_covisit_pivot):
        covisit = pl.read_parquet(
            f"data/candidates/{train_test}_covisit_top10_candidates.parquet")
        df_candidate = (df_candidate.join(
            covisit, how="left", left_on=[f"yad_no_{i+1}", "yad_no_cand"], right_on=["latest_yad_no", "yad_no"]).with_columns(
            pl.col("co_visit_count").cast(pl.Int16)).rename({"co_visit_count": f"co_visit_count_{i+1}_cand"})
        )

    # next seq feature
    next_seq = pl.read_parquet(
        f"data/candidates/{train_test}_next_seq_top10_candidates.parquet")
    df_candidate = (df_candidate.join(
        next_seq, how="left", left_on=["yad_no_1", "yad_no_cand"], right_on=["latest_yad_no", "yad_no"]).with_columns(
        pl.col("count").cast(pl.Int16)).rename({"count": "next_seq_count_1_cand"})
    )

    # make feature in discussion
    feature_name_list = [
        'latest_next_booking_top20',
        'past_view_yado',
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
        print(feature)

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

    df_candidate = df_candidate.with_columns([
        pl.col("latest_next_booking_rank").cast(pl.Int8),
        pl.col("max_seq_no").cast(pl.Int8),
        pl.col("max_seq_no_diff").cast(pl.Int8),
        pl.col("session_view_count").cast(pl.Int8),
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
        'total_room_cnt', 'wireless_lan_flg', 'onsen_flg', 'kd_stn_5min', 'kd_bch_5min', 'kd_slp_5min', 'kd_conv_walk_5min']

    pivot_dont_drop_cols = [
        "wid_cd_1", "ken_cd_1", "lrg_cd_1", 
    ]

    for i in range(n_pivot):
        for col in df_yad.columns:
            if col != "total_room_cnt" and f"{col}_{i+1}" not in pivot_dont_drop_cols and f"{col}_{i+1}" in df_candidate.columns:
                df_candidate = df_candidate.drop(f"{col}_{i+1}")

    for col in df_yad.columns:
        if col not in cand_dont_drop_cols and col != "yad_no" and f"{col}_cand" in df_candidate.columns:
            df_candidate = df_candidate.drop(f"{col}_cand")

    # df_candidate=df_candidate.drop(
    #     ["wid_cd", "ken_cd", "lrg_cd", "sml_cd", ])
    # change to category
    df_candidate = df_candidate.with_columns([
        pl.col("wid_cd").cast(pl.Categorical),
        pl.col("ken_cd").cast(pl.Categorical),
        pl.col("lrg_cd").cast(pl.Categorical),
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
    df_candidate = df_candidate.drop("sml_cd")

    if train_test == "train":
        df_candidate = df_candidate.drop("fold")

    print(df_candidate.columns)

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
    make_feature_and_label("20231215", "20231215")
