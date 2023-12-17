import gc

import polars as pl
from tqdm import tqdm

from tools import data_loader


def get_session_id_list(log):
    return log.group_by('session_id').head(1).select(['session_id'])


def make_candidate(fold_num=5):
    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    test_log = loader.load_test_log()
    yado = loader.load_yado()
    label = loader.load_cv_label()

    session_candidate_name_list = [
        'past_view_yado',
    ]

    yado_candidate_name_list = [
        'latest_next_booking_top20',
        # "covisit_top10",
        # "next_seq_top10"
    ]
    network_candidate_name_list = [
        # "network_top20",
    ]
    top10_area_candidate_name_list = [
        "sml",
        "lrg",
        "ken",
        # "wid"
    ]

    train_session_id = get_session_id_list(train_log)
    train_session_id = train_session_id.join(label.select(
        ['fold', 'session_id']), how='left', on='session_id')
    test_session_id = get_session_id_list(test_log)

    # session candidate
    for train_test in ['train', 'test']:
        candidate_list = []

        # for candidate_name in tqdm(cross_join_candidate_name_list):
        #     candidate = pl.read_parquet(
        #         f'data/candidates/{train_test}_{candidate_name}_candidates.parquet')
        #     if train_test == "train":
        #         candidate_all = pl.DataFrame()
        #         for fold in range(fold_num):
        #             candidate_fold = train_session_id.filter(pl.col('fold') == fold).join(
        #                 candidate.filter(pl.col('fold') == fold).select(['yad_no']), how='cross')
        #             candidate_all = pl.concat([candidate_all, candidate_fold])
        #         candidate_list.append(
        #             candidate_all.select(['session_id', 'yad_no']))
        #     else:
        #         candidate_all = candidate.join(
        #             test_session_id, how='cross', on="fold")
        #         candidate_list.append(
        #             candidate_all.select(['session_id', 'yad_no']))

        for candidate_name in tqdm(session_candidate_name_list):
            candidate = pl.read_parquet(
                f'data/candidates/{train_test}_{candidate_name}_candidates.parquet')
            candidate_list.append(
                candidate.select(['session_id', 'yad_no']))

        for candidate_name in tqdm(yado_candidate_name_list):
            candidate = pl.read_parquet(
                f'data/candidates/{train_test}_{candidate_name}_candidates.parquet')
            if train_test == 'train':
                latest_yad_no = train_log.group_by('session_id').tail(1).select(
                    ['session_id', 'yad_no']).rename({'yad_no': 'latest_yad_no'})
                if "fold" in candidate.columns:
                    latest_yad_no = latest_yad_no.join(label.select(
                        ['session_id', 'fold']), how='left', on='session_id')
                    latest_yad_no = latest_yad_no.with_columns(
                        pl.col('fold').cast(pl.Int32))
                    candidate = latest_yad_no.join(candidate, how='inner', on=[
                                                   'latest_yad_no', 'fold'])
                else:
                    candidate = latest_yad_no.join(
                        candidate, how='inner', on=['latest_yad_no'])
            else:
                latest_yad_no = test_log.group_by('session_id').tail(1).select(
                    ['session_id', 'yad_no']).rename({'yad_no': 'latest_yad_no'})
                candidate = latest_yad_no.join(
                    candidate, how='inner', on=['latest_yad_no'])

            candidate_list.append(
                candidate.select(['session_id', 'yad_no']))

        # network candidate
        for candidate_name in tqdm(network_candidate_name_list):

            candidate = pl.read_parquet(
                f'data/candidates/train_{candidate_name}_candidates.parquet')
            if train_test == 'train':
                latest_yad_no = train_log.group_by('session_id').tail(
                    1).select(["session_id", "yad_no"])
            else:
                latest_yad_no = test_log.group_by('session_id').tail(
                    1).select(["session_id", "yad_no"])

            candidate = latest_yad_no.join(
                candidate, how="inner", on=["yad_no"])

            candidate_list.append(candidate.select(['session_id', 'candidate_yad_no']).rename(
                {"candidate_yad_no": "yad_no"}))

        # area candidate
        for area_name in tqdm(top10_area_candidate_name_list):
            area_cd = area_name + '_cd'
            candidate = pl.read_parquet(
                f'data/candidates/{train_test}_top10_{area_name}_popular_yado_candidates.parquet')

            if train_test == 'train':
                less_than_15_session = train_session_id.join(pl.concat(candidate_list).unique(), on="session_id", how="left").fill_nan(0).group_by("session_id").count().filter(pl.col("count") < 10).select(["session_id"])
                candidate_all = pl.DataFrame()
                for fold in range(fold_num):
                    train_areas = (train_log
                        .join(yado[["yad_no", area_cd]], on="yad_no", how="left")
                        .sort("seq_no", descending=True)
                        .join(label[["session_id", "fold"]], on="session_id", how="left")
                        .filter(pl.col('fold') == fold) 
                        .filter(pl.col("session_id").is_in(less_than_15_session["session_id"]) == True)
                        .unique(["session_id", area_cd])[["session_id", area_cd]])

                    candidate_fold = (train_areas
                        .join(candidate.filter(pl.col("fold") == fold), how="left", on=area_cd)
                        [["session_id", "yad_no"]]
                    )
                    candidate_all = pl.concat([candidate_all, candidate_fold])

            else:
                less_than_15_session = test_session_id.join(pl.concat(candidate_list).unique(), on="session_id", how="left").fill_nan(0).group_by("session_id").count().filter(pl.col("count")<10).select("session_id")
                test_areas = (test_log.join(yado.select(['yad_no', area_cd,]), how='left', on='yad_no')
                    .select( ['session_id', area_cd])
                    .filter(pl.col("session_id").is_in(less_than_15_session["session_id"]) == True)
                    .unique(["session_id", area_cd]))

                candidate_all = test_areas.join(
                    candidate, how='left', on=area_cd).select(["session_id", "yad_no"])

            candidate_list.append(
                candidate_all
            )
            del candidate_all, candidate
            gc.collect()

        candidate = pl.concat(candidate_list).unique()

        print(test_session_id.join(pl.concat(candidate_list).unique(), on="session_id").fill_nan(0).group_by("session_id").count().filter(pl.col("count") < 15).select(["session_id"]))
        print(candidate.group_by('session_id').count().sort( 'count',))

        print(candidate.shape)

        candidate.write_parquet(
            f'data/candidates/{train_test}_candidate_20231215.parquet')

        del candidate
        gc.collect()


if __name__ == '__main__':

    make_candidate()
