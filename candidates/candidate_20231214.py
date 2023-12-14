import gc

import polars as pl
from tqdm import tqdm

from tools import data_loader


def create_past_view_yado_candidates(log):
    """
    アクセスした宿をcandidateとして作成。ただし、直近の宿は予約しないので除外する。
    """
    max_seq_no = log.group_by("session_id").agg(
        pl.max("seq_no").alias("max_seq_no"))
    log = log.join(max_seq_no, on="session_id")
    past_yado_candidates = log.filter(pl.col("seq_no") != pl.col("max_seq_no"))
    past_yado_candidates = past_yado_candidates.select(
        ['session_id', 'yad_no']).unique()

    past_yado_feature = log.with_columns((pl.col('max_seq_no') - pl.col('seq_no')).alias(
        'max_seq_no_diff')).filter(pl.col("seq_no") != pl.col("max_seq_no"))
    # 最後に見たseq
    past_yado_feature = past_yado_feature.join(past_yado_feature.group_by(["session_id", "yad_no"]).agg(
        pl.col("max_seq_no_diff").max().alias("max_seq_no_diff")), on=["session_id", "yad_no", "max_seq_no_diff"])
    # sessionの宿の数
    session_view_count = log.group_by(['session_id', 'yad_no']).count().rename({
        'count': 'session_view_count'})
    past_yado_feature = past_yado_feature.join(session_view_count, how='left', on=[
                                               'session_id', 'yad_no']).drop('seq_no')

    return past_yado_candidates, past_yado_feature


def make_candidate(train_log, test_log, label, yado, fold_num=5):

    train_past_view_yado_candidates, train_past_view_yado_feature = create_past_view_yado_candidates(
        train_log)
    test_past_view_yado_candidates, test_past_view_yado_feature = create_past_view_yado_candidates(
        test_log)

    # save all by .parquet
    train_past_view_yado_candidates.write_parquet(
        'data/candidates/train_past_view_yado_candidates.parquet')
    test_past_view_yado_candidates.write_parquet(
        'data/candidates/test_past_view_yado_candidates.parquet')


    train_past_view_yado_feature.write_parquet(
        'data/features/train_past_view_yado_feature.parquet')
    test_past_view_yado_feature.write_parquet(
        'data/features/test_past_view_yado_feature.parquet')


def get_session_id_list(log):
    return log.group_by('session_id').head(1).select(['session_id'])


def gen_candidate(fold_num=5):
    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    test_log = loader.load_test_log()
    yado = loader.load_yado()
    label = loader.load_cv_label()

    session_candidate_name_list = ['past_view_yado']

    yado_candidate_name_list = [
        'latest_next_booking_top20'
    ]
    top10_area_candidate_name_list = [
    ]

    # train_session_id = get_session_id_list(train_log)
    # train_session_id = train_session_id.join(label.select(
    #     ['fold', 'session_id']), how='left', on='session_id')
    # test_session_id = get_session_id_list(test_log)


    # session candidate
    for train_test in ['train', 'test']:
        candidate_list = []
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
                latest_yad_no = latest_yad_no.join(label.select(
                    ['session_id', 'fold']), how='left', on='session_id')
                latest_yad_no = latest_yad_no.with_columns(
                    pl.col('fold').cast(pl.Int32))
                candidate = latest_yad_no.join(candidate, how='inner', on=[
                                               'latest_yad_no', 'fold'])
            else:
                latest_yad_no = test_log.group_by('session_id').tail(1).select(
                    ['session_id', 'yad_no']).rename({'yad_no': 'latest_yad_no'})
                candidate = latest_yad_no.join(
                    candidate, how='inner', on=['latest_yad_no'])
            candidate_list.append(
                candidate.select(['session_id', 'yad_no']))


        # area candidate
        for area_name in tqdm(top10_area_candidate_name_list):
            area_cd = area_name + '_cd'
            candidate = pl.read_parquet(
                f'data/candidates/{train_test}_top10_{area_name}_popular_yado_candidates.parquet')
            if train_test == 'train':
                candidate_all = pl.DataFrame()
                for fold in range(fold_num):
                    train_areas = train_log.join(yado[["yad_no", area_cd]], on = "yad_no", how = "left").sort("seq_no", descending=True).join(label[["session_id", "fold"]], on="session_id", how="left").filter(pl.col('fold') == fold)
                    train_areas = train_areas.unique(["session_id", area_cd])[["session_id", area_cd]]
                    candidate_fold = train_areas.join(candidate, how="left", on=area_cd)[["session_id", "yad_no"]]
                    candidate_all = pl.concat([candidate_all, candidate_fold])

            else:
                test_areas = test_log.join(yado.select(['yad_no', area_cd,]), how='left', on='yad_no').select(
                    ['session_id', area_cd]).unique(["session_id", area_cd])
                candidate_all = test_areas.join(
                    candidate, how='left', on=area_cd).select(["session_id", "yad_no"])

            candidate_list.append(
                candidate_all
            )
            del candidate_all, candidate
            gc.collect()

        candidate = pl.concat(candidate_list).unique()

        candidate.write_parquet(
            f'data/candidates/{train_test}_candidate_20231214.parquet')

        del candidate
        gc.collect()


if __name__ == '__main__':
    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    test_log = loader.load_test_log()
    label = loader.load_cv_label()
    yado = loader.load_yado()

    make_candidate(train_log, test_log, label, yado)
    gen_candidate()
