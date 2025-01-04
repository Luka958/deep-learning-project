from datasets import load_dataset
from datasets import get_dataset_config_info
from pandas import DataFrame


def _get_row_count(config_name: str):
    dataset_info = get_dataset_config_info('BeIR/hotpotqa', config_name=config_name)
    
    return  dataset_info.splits[config_name].num_examples

def load_datasets(corpus_count: int, queries_count: int) -> tuple[DataFrame, DataFrame, DataFrame]:
    max_corpus_count = _get_row_count('corpus')
    max_query_count = _get_row_count('queries')
    
    if corpus_count > max_corpus_count or queries_count > max_query_count:
        raise ValueError()
    
    # load
    corpus = load_dataset('BeIR/hotpotqa', 'corpus', split=f'corpus[:{corpus_count}]')
    queries = load_dataset('BeIR/hotpotqa', 'queries', split=f'queries[:{queries_count}]')
    qrels = load_dataset('BeIR/hotpotqa-qrels')

    # filter
    query_ids_set = set(queries['_id'])
    corpus_ids_set = set(corpus['_id'])

    qrels_df = qrels['train'].to_pandas()
    filtered_qrels_df = qrels_df[
        qrels_df['corpus-id'].astype(str).isin(corpus_ids_set) &
        qrels_df['query-id'].isin(query_ids_set)
    ]

    unique_corpus_ids = set(filtered_qrels_df['corpus-id'].astype(str))
    unique_query_ids = set(filtered_qrels_df['query-id'])

    filtered_corpus = corpus.filter(lambda x: x['_id'] in unique_corpus_ids)
    filtered_queries = queries.filter(lambda x: x['_id'] in unique_query_ids)
    
    filtered_corpus_df = filtered_corpus.to_pandas()
    filtered_queries_df = filtered_queries.to_pandas()
    
    return filtered_corpus_df, filtered_queries_df, filtered_qrels_df