import time

import numpy as np
import pandas as pd
import polars as pl

from tools import data_loader



def get_features(feature_list, df_yad, df_log, df_candidate):
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

    pivoted = pivoted.select(["session_id", "1", "2", "3", "4"])
    df_candidate = df_candidate.join(pivoted, how="left", left_on="session_id", right_on="session_id", suffix="_cand")

    # pivoted = pivoted.join(df_candidate, how="cross",
    #                        left_on="session_id", right_on="session_id", suffix="_cand")

    for i in range(4):
        df_candidate = df_candidate.join(
            df_yad, how="left", left_on=f"{i+1}", right_on="yad_no", suffix=(f'_{i+1}'))

    df_candidate = df_candidate.join(
        df_yad, how="left", left_on="yad_no", right_on="yad_no", suffix="_cand")

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

    features_train = get_features(featuer_list, yado, train_log, candidate)
    features_test = get_features(featuer_list, yado, test_log, candidate_test)


      # if candidate["session_id", "yad_no"] is in train_label: then 1 else 0
    label_train = (
        candidate
        .select("session_id", "yad_no")
        .join(label_train, how="left", on="session_id")
        .select("session_id", "yad_no", "fold")
        .join(label_train, how="left", left_on=[
            "session_id", "yad_no"], right_on=["session_id", "yad_no"], suffix="_label")
        # .select(["session_id", "yad_no", "session_id_label"])
        .with_columns(pl.col("fold_label").is_not_null().cast(int).alias("label"))
        .select(["session_id", "yad_no", "fold", "label"])
    )

    label_train = features_train.select(["session_id", "yad_no"]).join(
        label_train, how="left", left_on=["session_id", "yad_no"], right_on=["session_id", "yad_no"],)


    timestamp = time.strftime("%Y%m%d%H%M%S")
    features_train.write_parquet(
        f'data/features/features_cand_train_{candidate_train_tstamp}_{timestamp}.parquet')
    features_test.write_parquet(
        f'data/features/features_cand_test_{candidate_test_tstamp}_{timestamp}.parquet')
    label_train.write_parquet(
        f'data/features/label_cand_{candidate_train_tstamp}_{timestamp}.parquet')


if __name__ == "__main__":
    make_feature_and_label("20231211", "20231211")
