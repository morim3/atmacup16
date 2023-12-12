import numpy as np
import polars as pl

from tools import data_loader

def make_location_embedding():
    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    yado = loader.load_yado()

    # wid_cd : 広域エリア CD wide area CD
    # ken_cd : 県 CD prefecture CD
    # lrg_cd : 大エリア CD large area CD
    # sml_cd : 小エリア CD small area CD
    yado_location = yado[["yado_no", "wid_cd", "ken_cd", "lrg_cd", "sml_cd"]]



