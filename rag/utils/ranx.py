from qdrant_client.http.models import ScoredPoint
from pandas import DataFrame
from ranx import Qrels, Run


def get_qrels(qrels_df: DataFrame) -> Qrels:
    qrels_dict = {}

    for _, row in qrels_df.iterrows():
        query_id = row['query-id']
        corpus_id = str(row['corpus-id'])
        relevance = int(row['score'])
        
        if query_id not in qrels_dict:
            qrels_dict[query_id] = {}

        qrels_dict[query_id][corpus_id] = relevance
        
    return Qrels(qrels_dict)


def get_run(queries_df: DataFrame, scored_points_list: list[list[ScoredPoint]]) -> Run:
    runs_dict = {}

    for i, query_id in enumerate(queries_df['_id'].values):
        runs_dict[query_id] = {}
        
        for scored_point in scored_points_list[i]:
            doc_id = str(scored_point.payload['id'])
            runs_dict[query_id][doc_id] = float(scored_point.score)
            
    return Run(runs_dict)