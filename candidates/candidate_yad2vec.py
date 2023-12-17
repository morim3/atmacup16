import gc

import polars as pl
from tqdm import tqdm

from tools import data_loader
import pickle


def create_yad2vec_candidate(log, label=None, train_test="train", top=10):

    # load word2vec models
    with open("data/item2vec/yad2vec_models.pickle", "rb") as f:
        yad2vec_models = pickle.load(f)

    if train_test == "train":
        # care cv fold
        log = log.join(label, on="session_id", how="inner")

        session_ids = log.filter(pl.col("fold") != 0)[
            ["session_id", "fold"]].unique()

        cand_all = []
        for sess, fold in tqdm(zip(session_ids["session_id"], session_ids["fold"]), total=session_ids.shape[0]):

            candidate_sess = []

            for yado in log.filter(pl.col("session_id") == sess)["yad_no"].unique().to_list():

                if yado in yad2vec_models[fold].wv:
                    candidate_yado = yad2vec_models[fold].wv.most_similar(
                        yado, topn=top)
                    for w, _ in candidate_yado:
                        if w != yado:
                            candidate_sess.append(w)

            candidate_sess = pl.DataFrame({
                "session_id": [sess] * len(candidate_sess),
                "yad_no": candidate_sess
            })

            candidate_sess = candidate_sess.unique()
            cand_all.append(candidate_sess)

        cand_all = pl.concat(cand_all)

    else:
        session_ids = log["session_id"].unique()

        cand_all = []

        for sess in tqdm(session_ids):
            candidate_sess = []
            for yado in log.filter(pl.col("session_id") == sess)["yad_no"].unique().to_list():
                for fold in range(5):
                    if yado in yad2vec_models[fold].wv:
                        candidate_yado = yad2vec_models[fold].wv.most_similar(
                            yado, topn=top)
                        for w, _ in candidate_yado:
                            if w != yado:
                                candidate_sess.append(w)

            candidate_sess = pl.DataFrame({
                "session_id": [sess] * len(candidate_sess),
                "yad_no": candidate_sess
            })

            candidate_sess = candidate_sess.unique()
            cand_all.append(candidate_sess)

        cand_all = pl.concat(cand_all)

    return cand_all


if __name__ == '__main__':
    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    test_log = loader.load_test_log()
    label = loader.load_cv_label()
    yado = loader.load_yado()

    yad2vec_train_cand = create_yad2vec_candidate(
        train_log, label, "train", top=10)
    yad2vec_test_cand = create_yad2vec_candidate(
        train_log, label, "test", top=10)

    print(yad2vec_train_cand)
    print(yad2vec_test_cand)

    yad2vec_train_cand.write_parquet(
        "data/candidates/yad2vec_train_cand.parquet")
    yad2vec_test_cand.write_parquet(
        "data/candidates/yad2vec_test_cand.parquet")
