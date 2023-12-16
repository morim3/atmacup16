import polars as pl
from tools.data_loader import AtmaData16Loader


def get_candidate_next_seq(log, top=10):
    next_seq_log = log.with_columns(pl.col(
        'seq_no')+1)[['session_id', 'seq_no', 'yad_no']].rename({'yad_no': 'next_yad_no'})
    log_next_seq = log.join(next_seq_log, how='left', on=[
                            'session_id', 'seq_no']).filter(pl.col('next_yad_no').is_not_null())
    log_next_seq_feature = (log_next_seq.group_by(['yad_no', 'next_yad_no']).count().
                            sort(by=['yad_no', 'count'], descending=[False, True]))

    log_next_seq_candidate = (log_next_seq_feature.group_by('yad_no').head(
        top).rename({'yad_no': 'latest_yad_no', 'next_yad_no': 'yad_no'}))

    return log_next_seq_candidate , log_next_seq_feature


if __name__ == '__main__':
    loader = AtmaData16Loader()
    train_log = loader.load_train_log()
    test_log = loader.load_test_log()
    # train_cand_next_seq = get_candidate_next_seq(train_log)
    train_cand_next_seq, train_cand_next_seq_feature = get_candidate_next_seq(train_log)
    test_cand_next_seq, test_cand_next_seq_feature = get_candidate_next_seq(test_log)


    # latest_yad_no, yad_no, count
    train_cand_next_seq.drop('count').write_parquet(
        'data/candidates/train_next_seq_top10_candidates.parquet')
    test_cand_next_seq.drop("count").write_parquet(
        'data/candidates/test_next_seq_top10_candidates.parquet')

    train_cand_next_seq_feature.write_parquet(
        'data/features/train_next_seq_features.parquet')
    test_cand_next_seq_feature.write_parquet(
        'data/features/test_next_seq_features.parquet')
