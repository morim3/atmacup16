import polars as pl


def create_past_view_yado_candidates(log, candidates):
    """
    アクセスした宿をcandidateとして作成。ただし、直近の宿は予約しないので除外する。
    """

    past_yado_feature = log.with_columns((pl.col('max_seq_no') - pl.col('seq_no')).alias(
        'max_seq_no_diff')).filter(pl.col("seq_no") != pl.col("max_seq_no"))
    past_yado_feature = past_yado_feature.join(past_yado_feature.group_by(["session_id", "yad_no"]).agg(
        pl.col("max_seq_no_diff").max().alias("max_seq_no_diff")), on=["session_id", "yad_no", "max_seq_no_diff"])
    session_view_count = log.group_by(['session_id', 'yad_no']).count().rename({
        'count': 'session_view_count'})
    past_yado_feature = past_yado_feature.join(session_view_count, how='left', on=[
                                               'session_id', 'yad_no']).drop('seq_no')


    return past_yado_feature


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
