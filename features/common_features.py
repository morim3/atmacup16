import numpy as np
import polars as pl




def session_mean_length_each_yad(log, ):

    unique_yad = log["yad_no"].unique().to_list()

    session_len = log.groupby("session_id").max("seq_no")

    mean_length = []
    for yad in unique_yad:
        mean_length.append(log.filter(pl.col("yad_no") == yad)






