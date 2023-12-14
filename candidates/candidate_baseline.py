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
    past_yado_feature = past_yado_feature.join(past_yado_feature.group_by(["session_id", "yad_no"]).agg(
        pl.col("max_seq_no_diff").max().alias("max_seq_no_diff")), on=["session_id", "yad_no", "max_seq_no_diff"])
    session_view_count = log.group_by(['session_id', 'yad_no']).count().rename({
        'count': 'session_view_count'})
    past_yado_feature = past_yado_feature.join(session_view_count, how='left', on=[
                                               'session_id', 'yad_no']).drop('seq_no')

    return past_yado_candidates, past_yado_feature


def create_topN_popular_yado_candidates(label, fold_num, train_test='train', top=10):
    """
    予約された人気宿をcandidateとして作成。train/validでリークしないように注意。
    """
    # labelデータを使うので、学習データはtrain/validで分割して作成。
    top10_yado_candidate = pl.DataFrame()
    popular_yado_feature = pl.DataFrame()
    if train_test == 'train':
        for fold in range(fold_num):
            train_label = label.filter(pl.col('fold') != fold)
            popular_yado_sort = train_label['yad_no'].value_counts().sort(
                by='counts', descending=True)

            # candidateの作成
            top10_yado_candidate_fold = popular_yado_sort.head(top).with_columns(
                pl.lit(fold).alias('fold')).select(['yad_no', 'fold'])
            top10_yado_candidate = pl.concat(
                [top10_yado_candidate, top10_yado_candidate_fold])

            # 簡易的な特徴量も作成しておく。
            popular_yado_feature_fold = popular_yado_sort.with_columns(
                pl.lit(fold).alias('fold'))
            popular_yado_feature_fold = popular_yado_feature_fold.with_columns(
                pl.arange(1, len(popular_yado_sort)+1).alias('popular_rank'))
            popular_yado_feature = pl.concat(
                [popular_yado_feature, popular_yado_feature_fold])
    else:
        # candidateの作成
        popular_yado_sort = label['yad_no'].value_counts().sort(
            by='counts', descending=True)
        top10_yado_candidate = popular_yado_sort.head(top).select(['yad_no'])

        # 簡易的な特徴量も作成しておく。
        popular_yado_feature = popular_yado_sort.with_columns(
            pl.arange(1, len(popular_yado_sort)+1).alias('popular_rank'))

    popular_yado_feature = popular_yado_feature.rename(
        {'counts': 'reservation_counts'})

    return top10_yado_candidate, popular_yado_feature


def create_topN_area_popular_yado_candidates(label, yado, fold_num, train_test='train', area='wid_cd', top=10):
    """
    Returns:
        pl.DataFrame: area, yad_no
    Note:
        エリア単位で予約された人気宿をcandidateとして作成。train/validでリークしないように注意。
    """
    # labelデータを使うので、学習データはtrain/validで分割して作成。
    top10_yado_area_candidate = pl.DataFrame()
    popular_yado_area_feature = pl.DataFrame()
    if train_test == 'train':
        for fold in range(fold_num):
            train_label = label.filter(pl.col('fold') != fold)
            label_yado = train_label.join(yado, how='left', on='yad_no')
            popular_yado_sort = label_yado.group_by([area, 'yad_no']).count().sort(
                by=[area, 'count'], descending=[False, True])

            # candidateの作成
            top10_yado_area_candidate_fold = popular_yado_sort.group_by(area).head(
                top).with_columns(pl.lit(fold).alias('fold')).select([area, 'yad_no', 'fold'])
            top10_yado_area_candidate = pl.concat(
                [top10_yado_area_candidate, top10_yado_area_candidate_fold])

            # 簡易的な特徴量も作成しておく。
            popular_yado_area_feature_fold = popular_yado_sort.with_columns(
                pl.lit(fold).alias('fold'))
            popular_yado_area_feature_fold = (popular_yado_area_feature_fold
                                              .group_by(area)
                                              .map_groups(lambda group: group.with_columns(pl.col('count').rank(method='dense', descending=True).over(area).alias(f'popular_{area}_rank'))))
            popular_yado_area_feature = pl.concat(
                [popular_yado_area_feature, popular_yado_area_feature_fold])

    else:  # testデータはtrainデータ全体で作成する。
        label_yado = label.join(yado, how='left', on='yad_no')
        popular_yado_sort = label_yado.group_by([area, 'yad_no']).count().sort(
            by=[area, 'count'], descending=[False, True])
        top10_yado_area_candidate = popular_yado_sort.group_by(
            area).head(top).select([area, 'yad_no'])

        # 簡易的な特徴量も作成しておく。
        popular_yado_area_feature = (popular_yado_sort
                                     .group_by(area)
                                     .map_groups(lambda group: group.with_columns(pl.col('count').rank(method='dense', descending=True).over(area).alias(f'popular_{area}_rank'))))

    popular_yado_area_feature = popular_yado_area_feature.drop('count')

    return top10_yado_area_candidate, popular_yado_area_feature


def create_latest_next_booking_topN_candidate(train_log, label, fold_num, train_test='train', top=10):
    """
    直近見た宿で、次にどこを予約しやすいか。
    """
    log_latest = train_log.group_by('session_id').tail(1)
    log_latest = log_latest.rename({'yad_no': 'latest_yad_no'})
    log_latest = log_latest.join(label, how='left', on='session_id')

    # labelデータを使うので、学習データはtrain/validで分割して作成。
    latest_next_booking_topN_candidate = pl.DataFrame()
    latest_next_booking_topN_feature = pl.DataFrame()
    if train_test == 'train':
        for fold in range(fold_num):
            train_log_latest = log_latest.filter(pl.col('fold') != fold)
            train_log_latest = train_log_latest.group_by(['latest_yad_no', 'yad_no']).count(
            ).sort(by=['latest_yad_no', 'count'], descending=[False, True])

            # candidateの作成
            latest_next_booking_topN_candidate_fold = train_log_latest.group_by('latest_yad_no').head(
                top).with_columns(pl.lit(fold).alias('fold')).select(['yad_no', 'latest_yad_no', 'fold'])
            latest_next_booking_topN_candidate = pl.concat(
                [latest_next_booking_topN_candidate, latest_next_booking_topN_candidate_fold])

            # 簡易的な特徴量も作成しておく。
            latest_next_booking_topN_feature_fold = train_log_latest.with_columns(
                pl.lit(fold).alias('fold'))
            latest_next_booking_topN_feature_fold = (latest_next_booking_topN_feature_fold
                                                     .group_by('latest_yad_no')
                                                     .map_groups(lambda group: group.with_columns(pl.col('count').rank(method='dense', descending=True).over('latest_yad_no').alias(f'latest_next_booking_rank'))))
            latest_next_booking_topN_feature = pl.concat(
                [latest_next_booking_topN_feature, latest_next_booking_topN_feature_fold])
    else:
        log_latest = log_latest.group_by(['latest_yad_no', 'yad_no']).count().sort(
            by=['latest_yad_no', 'count'], descending=[False, True])

        # candidateの作成
        latest_next_booking_topN_candidate = log_latest.group_by(
            'latest_yad_no').head(top).select(['yad_no', 'latest_yad_no'])

        # 簡易的な特徴量も作成しておく。
        latest_next_booking_topN_feature = (log_latest
                                            .group_by('latest_yad_no')
                                            .map_groups(lambda group: group.with_columns(pl.col('count').rank(method='dense', descending=True).over('latest_yad_no').alias(f'latest_next_booking_rank'))))
    latest_next_booking_topN_feature = latest_next_booking_topN_feature.drop(
        'count')
    return latest_next_booking_topN_candidate, latest_next_booking_topN_feature


def make_candidate(train_log, test_log, label, yado, fold_num=5):

    train_past_view_yado_candidates, train_past_view_yado_feature = create_past_view_yado_candidates(
        train_log)
    test_past_view_yado_candidates, test_past_view_yado_feature = create_past_view_yado_candidates(
        test_log)
    train_top20_popular_yado_candidates, train_top20_popular_yado_feature = create_topN_popular_yado_candidates(
        label, fold_num=fold_num, train_test='train', top=20)
    test_top20_popular_yado_candidates, test_top20_popular_yado_feature = create_topN_popular_yado_candidates(
        label, fold_num=fold_num, train_test='test', top=20)

    train_top10_wid_popular_yado_candidates, train_top10_wid_popular_yado_feature = create_topN_area_popular_yado_candidates(
        label, yado, fold_num=fold_num, train_test='train', area='wid_cd', top=10)
    test_top10_wid_popular_yado_candidates, test_top10_wid_popular_yado_feature = create_topN_area_popular_yado_candidates(
        label, yado, fold_num=fold_num, train_test='test', area='wid_cd', top=10)

    train_top10_ken_popular_yado_candidates, train_top10_ken_popular_yado_feature = create_topN_area_popular_yado_candidates(
        label, yado, fold_num=fold_num, train_test='train', area='ken_cd', top=10)
    test_top10_ken_popular_yado_candidates, test_top10_ken_popular_yado_feature = create_topN_area_popular_yado_candidates(
        label, yado, fold_num=fold_num, train_test='test', area='ken_cd', top=10)

    train_top10_lrg_popular_yado_candidates, train_top10_lrg_popular_yado_feature = create_topN_area_popular_yado_candidates(
        label, yado, fold_num=fold_num, train_test='train', area='lrg_cd', top=10)

    test_top10_lrg_popular_yado_candidates, test_top10_lrg_popular_yado_feature = create_topN_area_popular_yado_candidates(
        label, yado, fold_num=fold_num, train_test='test', area='lrg_cd', top=10)

    train_top10_sml_popular_yado_candidates, train_top10_sml_popular_yado_feature = create_topN_area_popular_yado_candidates(
        label, yado, fold_num=fold_num, train_test='train', area='sml_cd', top=10)
    test_top10_sml_popular_yado_candidates, test_top10_sml_popular_yado_feature = create_topN_area_popular_yado_candidates(
        label, yado, fold_num=fold_num, train_test='test', area='sml_cd', top=10)
    train_latest_next_booking_top20_candidate, train_latest_next_booking_top20_feature = create_latest_next_booking_topN_candidate(
        train_log, label, fold_num=fold_num, train_test='train', top=20)
    test_latest_next_booking_top20_candidate, test_latest_next_booking_top20_feature = create_latest_next_booking_topN_candidate(
        train_log, label, fold_num=fold_num, train_test='test', top=20)

    # save all by .parquet
    train_past_view_yado_candidates.write_parquet(
        'data/candidates/train_past_view_yado_candidates.parquet')
    test_past_view_yado_candidates.write_parquet(
        'data/candidates/test_past_view_yado_candidates.parquet')
    train_top20_popular_yado_candidates.write_parquet(
        'data/candidates/train_top20_popular_yado_candidates.parquet')
    test_top20_popular_yado_candidates.write_parquet(
        'data/candidates/test_top20_popular_yado_candidates.parquet')
    train_top10_wid_popular_yado_candidates.write_parquet(
        'data/candidates/train_top10_wid_popular_yado_candidates.parquet')
    test_top10_wid_popular_yado_candidates.write_parquet(
        'data/candidates/test_top10_wid_popular_yado_candidates.parquet')
    train_top10_ken_popular_yado_candidates.write_parquet(
        'data/candidates/train_top10_ken_popular_yado_candidates.parquet')
    test_top10_ken_popular_yado_candidates.write_parquet(
        'data/candidates/test_top10_ken_popular_yado_candidates.parquet')
    train_top10_lrg_popular_yado_candidates.write_parquet(
        'data/candidates/train_top10_lrg_popular_yado_candidates.parquet')
    test_top10_lrg_popular_yado_candidates.write_parquet(
        'data/candidates/test_top10_lrg_popular_yado_candidates.parquet')
    train_top10_sml_popular_yado_candidates.write_parquet(
        'data/candidates/train_top10_sml_popular_yado_candidates.parquet')
    test_top10_sml_popular_yado_candidates.write_parquet(
        'data/candidates/test_top10_sml_popular_yado_candidates.parquet')
    train_latest_next_booking_top20_candidate.write_parquet(
        'data/candidates/train_latest_next_booking_top20_candidates.parquet')
    test_latest_next_booking_top20_candidate.write_parquet(
        'data/candidates/test_latest_next_booking_top20_candidates.parquet')

    train_past_view_yado_feature.write_parquet(
        'data/features/train_past_view_yado_feature.parquet')
    test_past_view_yado_feature.write_parquet(
        'data/features/test_past_view_yado_feature.parquet')
    train_top20_popular_yado_feature.write_parquet(
        'data/features/train_top20_popular_yado_feature.parquet')
    test_top20_popular_yado_feature.write_parquet(
        'data/features/test_top20_popular_yado_feature.parquet')
    train_top10_wid_popular_yado_feature.write_parquet(
        'data/features/train_top10_wid_popular_yado_feature.parquet')
    test_top10_wid_popular_yado_feature.write_parquet(
        'data/features/test_top10_wid_popular_yado_feature.parquet')
    train_top10_ken_popular_yado_feature.write_parquet(
        'data/features/train_top10_ken_popular_yado_feature.parquet')
    test_top10_ken_popular_yado_feature.write_parquet(
        'data/features/test_top10_ken_popular_yado_feature.parquet')
    train_top10_lrg_popular_yado_feature.write_parquet(
        'data/features/train_top10_lrg_popular_yado_feature.parquet')
    test_top10_lrg_popular_yado_feature.write_parquet(
        'data/features/test_top10_lrg_popular_yado_feature.parquet')
    train_top10_sml_popular_yado_feature.write_parquet(
        'data/features/train_top10_sml_popular_yado_feature.parquet')
    test_top10_sml_popular_yado_feature.write_parquet(
        'data/features/test_top10_sml_popular_yado_feature.parquet')
    train_latest_next_booking_top20_feature.write_parquet(
        'data/features/train_latest_next_booking_top20_feature.parquet')
    test_latest_next_booking_top20_feature.write_parquet(
        'data/features/test_latest_next_booking_top20_feature.parquet')


def get_session_id_list(log):
    return log.group_by('session_id').head(1).select(['session_id'])


def candidate_20231211(fold_num=5):
    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    test_log = loader.load_test_log()
    yado = loader.load_yado()
    label = loader.load_cv_label()

    cross_join_candidate_name_list = [
        'top20_popular_yado',
    ]
    session_candidate_name_list = ['past_view_yado',
                                   ]

    yado_candidate_name_list = [
        'latest_next_booking_top20'
    ]
    top10_area_candidate_name_list = [
        # "sml",
        # "lrg",
        # "ken"
        # "wid"
    ]

    train_session_id = get_session_id_list(train_log)
    train_session_id = train_session_id.join(label.select(
        ['fold', 'session_id']), how='left', on='session_id')
    test_session_id = get_session_id_list(test_log)

    # session candidate
    for train_test in ['train', 'test']:
        candidate_list = []
        for candidate_name in tqdm(cross_join_candidate_name_list):
            candidate = pl.read_parquet(
                f'data/candidates/{train_test}_{candidate_name}_candidates.parquet')
            if train_test == "train":
                candidate_all = pl.DataFrame()
                for fold in range(fold_num):
                    candidate_fold = train_session_id.filter(pl.col('fold') == fold).join(candidate.filter(pl.col('fold') == fold).select(['yad_no']),how='cross')
                    candidate_all = pl.concat([candidate_all,candidate_fold])
                candidate_list.append(candidate_all.select(['session_id', 'yad_no']))
            else:
                candidate_all = candidate.join(
                    train_session_id, how='cross', on="fold")
                candidate_list.append(candidate_all.select(['session_id', 'yad_no']))

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
                    train_areas = train_log.join(yado[["yad_no", area_cd]], on="yad_no", how="left").sort("seq_no", descending=True).join(
                        label[["session_id", "fold"]], on="session_id", how="left").filter(pl.col('fold') == fold)
                    train_areas = train_areas.unique(["session_id", area_cd])[
                        ["session_id", area_cd]]
                    candidate_fold = train_areas.join(candidate, how="left", on=area_cd)[
                        ["session_id", "yad_no"]]
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
            f'data/candidates/{train_test}_candidate_baseline.parquet')

        del candidate
        gc.collect()


if __name__ == '__main__':
    loader = data_loader.AtmaData16Loader()
    train_log = loader.load_train_log()
    test_log = loader.load_test_log()
    label = loader.load_cv_label()
    yado = loader.load_yado()

    make_candidate(train_log, test_log, label, yado)
    candidate_20231211()
