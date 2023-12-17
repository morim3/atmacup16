import polars as pl
from tqdm import tqdm
import numpy as np


def generate_co_visit_matrix(df:pl.DataFrame) -> pl.DataFrame:
    # 共起ペアの作成
    df = df.join(df, on="session_id")
    # yad_noが同じものは除外する
    df = df.filter(pl.col("yad_no") != pl.col("yad_no_right"))

    df = df.group_by(["yad_no", "yad_no_right"]).count()

    df = df.rename(
        {
            "yad_no_right":"candidate_yad_no",
            "count":"co_visit_count",
        }
    )[["yad_no", "candidate_yad_no", "co_visit_count"]]

    return df



def create_covisitation_candidates(train_log, test_log, label, top=20, test_train="train"):
    """
    make pairs in session and count how many times they appear together
    then, top 20 pairs are returned
    """

    if test_train == "train":

        log = pl.concat([train_log, test_log])
        coappear = (log
                    .pivot(index="session_id", columns="seq_no", values="yad_no")
                    .join(label, on="session_id")
                    .to_numpy()
                    )

        appear_data = [[a for a in sess if not np.isnan(
            a)] for sess in coappear[:, 1:-1]]
        fold = coappear[:, -1]
        appear_data = [[int(a) for a in set(sess)]
                       for sess in appear_data if len(set(sess)) > 1]


        cand_all = []
        for f in range(5):

            coappaer_count = {}

            for i_sess, yados in enumerate(appear_data):
                coappaer_count[i_sess] = {}

            for i_sess, yados in enumerate(appear_data):
                if fold[i] != f:
                    for i in range(len(yados)):
                        for j in range(i + 1, len(yados)):
                            if yados[j] in coappaer_count[yados[i]]:
                                coappaer_count[yados[i]][yados[j]] += 1
                                coappaer_count[yados[j]][yados[i]] += 1

                            else:
                                coappaer_count[yados[i]][yados[j]] = 1
                                coappaer_count[yados[j]][yados[i]] = 1

            # sort by count
            for yado in coappaer_count:
                coappaer_count[yado] = dict(
                    sorted(coappaer_count[yado].items(), key=lambda x: x[1], reverse=True))

            # for each yado, get top 20 yados
            # the resulting data is like
            # session_id, yad_no

            unique_sess = log.filter(pl.col("fold") == f)["session_id"].unique()

            for sess in tqdm(unique_sess):
                yados = log.filter(pl.col("session_id") == sess)[
                    "yad_no"].unique().to_list()
                cand_sess = []
                for yado in yados:
                    if yado in coappaer_count:
                        cand_sess += list(coappaer_count[yado].keys())[:top]

                cand_sess = pl.DataFrame({
                    "session_id": [sess] * len(cand_sess),
                    "yad_no": [c[0] for c in cand_sess]
                })

                cand_all.append(cand_sess)


        cand_all = pl.concat(cand_all)
        cand_all = cand_all.unique()

        return cand_all

    else:
        coappear = (log
                    .pivot(index="session_id", columns="seq_no", values="yad_no")
                    .to_numpy()
                    )

        appear_data = [[a for a in sess if not np.isnan(
            a)] for sess in coappear[:, 1:]]
        appear_data = [[int(a) for a in set(sess)]
                       for sess in appear_data if len(set(sess)) > 1]

        coappaer_count = {}

        for i_sess, yados in enumerate(appear_data):
            coappaer_count[i_sess] = {}

        for i_sess, yados in enumerate(appear_data):
            for i in range(len(yados)):
                for j in range(i + 1, len(yados)):
                    if yados[j] in coappaer_count[yados[i]]:
                        coappaer_count[yados[i]][yados[j]] += 1
                        coappaer_count[yados[j]][yados[i]] += 1

                    else:
                        coappaer_count[yados[i]][yados[j]] = 1
                        coappaer_count[yados[j]][yados[i]] = 1

        # sort by count
        for yado in coappaer_count:
            coappaer_count[yado] = dict(
                sorted(coappaer_count[yado].items(), key=lambda x: x[1], reverse=True))

        unique_sess = log["session_id"].unique()

        cand_all = []
        for sess in tqdm(unique_sess):
            yados = log.filter(pl.col("session_id") == sess)[
                "yad_no"].unique().to_list()
            cand_sess = []
            for yado in yados:
                if yado in coappaer_count:
                    cand_sess += list(coappaer_count[yado].keys())[:top]

            cand_sess = pl.DataFrame({
                "session_id": [sess] * len(cand_sess),
                "yad_no": [c[0] for c in cand_sess]
            })

            cand_all.append(cand_sess)

        cand_all = pl.concat(cand_all)
        cand_all = cand_all.unique()

        return cand_all



if __name__ == "__main__":
    train_log = pl.read_parquet("data/train_log.parquet")
    test_log = pl.read_parquet("data/test_log.parquet")
    # label = pl.read_parquet("data/cv_label.parquet")

    top = 10


    train_co_visit_matrix = generate_co_visit_matrix(train_log)
    test_co_visit_matrix = generate_co_visit_matrix(test_log)


    train_co_visit_matrix = train_co_visit_matrix.rename({'yad_no':'latest_yad_no','candidate_yad_no':'yad_no'})
    test_co_visit_matrix = test_co_visit_matrix.rename({'yad_no':'latest_yad_no','candidate_yad_no':'yad_no'})

    train_co_visit_matrix_top10_candidate = train_co_visit_matrix.sort(['latest_yad_no','co_visit_count'],descending=[False,True]).group_by('latest_yad_no').head(20)
    test_co_visit_matrix_top10_candidate = test_co_visit_matrix.sort(['latest_yad_no','co_visit_count'],descending=[False,True]).group_by('latest_yad_no').head(20)

    print(train_co_visit_matrix_top10_candidate)

    print(train_co_visit_matrix_top10_candidate.group_by("latest_yad_no").count().sort("count",).head(10))

    #normalize by ranking
    train_co_visit_matrix = (
        train_co_visit_matrix
        .sort(['co_visit_count'],descending=[False])
        .with_columns(
               (pl.col("co_visit_count").rank(method="ordinal") * 127 / train_co_visit_matrix.shape[0]).cast(pl.UInt8, strict=False)
        )

    )

    test_co_visit_matrix = (
        test_co_visit_matrix
        .sort(['co_visit_count'],descending=[False])
        .with_columns(
                (pl.col("co_visit_count").rank(method="ordinal") * 127 / test_co_visit_matrix.shape[0]).cast(pl.UInt8, strict=False)
        )
    )

    print(train_co_visit_matrix)
        

    print(train_co_visit_matrix.to_pandas().describe())
    print(test_co_visit_matrix.to_pandas().describe())

    train_co_visit_matrix.write_parquet('data/features/train_covisit_features.parquet')
    test_co_visit_matrix.write_parquet('data/features/test_covisit_features.parquet')
    train_co_visit_matrix_top10_candidate.write_parquet('data/candidates/train_covisit_top10_candidates.parquet')
    test_co_visit_matrix_top10_candidate.write_parquet('data/candidates/test_covisit_top10_candidates.parquet')

    # cand_all_train = create_covisitation_candidates(
    #     log, label, top=top, train_test="train")
    # cand_all_test = create_covisitation_candidates(
    #     log, label, top=top, train_test="test")
    #
    # cand_all_train.write_parquet(
    #     f"data/candidates/covisitation_train_cand_top{top}.parquet")
    # cand_all_test.write_parquet(
    #     f"data/candidates/covisitation_test_cand_top{top}.parquet")
