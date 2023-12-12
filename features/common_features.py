import numpy as np
import polars as pl




def make_matching_features(matched_yad_id, df_yad):

    """
    matched_yad_id: pl.DataFrame
        session_id, yad_no_0, yad_no_1, ..., yad_no_9, yad_no_cand
    """
    # how many times candidate yado appered in yad_no_0-9


    # how many times candidate yado has same following features
    # yad_type : 宿泊種別 lodging type
    # total_room_cnt : 部屋数 total number of rooms
    # wireless_lan_flg : 無線 LAN があるかどうか wireless LAN connection
    # onsen_flg : 温泉を有しているかどうか flag with hot spring
    # kd_stn_5min : 駅まで 5 分以内かどうか within 5 minutes walk from the station
    # kd_bch_5min : ビーチまで 5 分以内かどうか within 5 minutes walk to the beach
    # kd_slp_5min : ゲレンデまで 5 分以内かどうか within 5 minutes walk to the slopes
    # kd_conv_walk_5min : コンビニまで 5 分以内かどうか within 5 minutes walk to convenience store
    features = ["yado_no", "yado_type", "total_room_cnt", "wireless_lan_flg", "onsen_flg", "kd_stn_5min", "kd_bch_5min", "kd_slp_5min", "kd_conv_walk_5min"]
    for f in features:
        for i in range(10):
            matched_yad_id = matched_yad_id.with_columns(
                pl.when(pl.col(f"{f}_{i}") == pl.col(f"{f}_cand")).then(1).otherwise(0).alias(f"{f}_{i}_cnt"),
            )

        matched_yad_id = matched_yad_id.with_columns(
            pl.col(f"{f}_cnt").sum().over("session_id").alias(f"{f}_cnt")
        )

    return matched_yad_id.select(
        [
            "session_id",
            "yad_no_yado_no_cnt",
            "yad_no_yado_type_cnt",
            "yad_no_total_room_cnt_cnt",
            "yad_no_wireless_lan_flg_cnt",
            "yad_no_onsen_flg_cnt",
            "yad_no_kd_stn_5min_cnt",
            "yad_no_kd_bch_5min_cnt",
            "yad_no_kd_slp_5min_cnt",
            "yad_no_kd_conv_walk_5min_cnt",
        ]
            



