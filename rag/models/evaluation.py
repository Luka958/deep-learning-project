from fastembed import TextEmbedding
from pandas import DataFrame
from ranx import Qrels, Run, evaluate

from .metadata import Metadata
from .search import DenseSearchManager


class Evaluator:
    def __init__(
        self,
        manager: DenseSearchManager,
        dense_model: TextEmbedding,
        collection_name: str,
        dfs: tuple[DataFrame, DataFrame, DataFrame],
    ):
        self.manager = manager
        self.dense_model = dense_model
        self.collection_name = collection_name
        self.corpus_df, self.queries_df, self.qrels_df = dfs
        
        self.qrels_dict = {}

        for _, row in self.qrels_df.iterrows():
            query_id = row['query-id']
            corpus_id = str(row['corpus-id'])
            relevance = int(row['score'])
            
            if query_id not in self.qrels_dict:
                self.qrels_dict[query_id] = {}

            self.qrels_dict[query_id][corpus_id] = relevance
    
    def setup(self) -> bool:
        # embed
        corpus_texts: list[str] = self.corpus_df['text'].values.tolist()
        metadatas = [
            Metadata(
                id=row['_id'],
                text=row['text']
            )
            for _, row in self.corpus_df.iterrows()
        ]

        dense_embeddings = list(self.dense_model.embed(corpus_texts))
        #sparse_embeddings = list(sparse_model.embed(corpus_texts)) # TODO
        #reranking_embeddings = list(reranking_model.embed(corpus_texts))
        
        # index
        self.manager.create_collection(self.collection_name)
        
        return self.manager.upload_points(
            self.collection_name,
            dense_embeddings,
            metadatas
        )
    
    def run(
        self,
        top_k: int,
        metrics: list[str]
    ) -> dict[str, float] | float:
        # embed
        query_texts: list[str] = self.queries_df['text'].values.tolist()
        query_dense_embeddings = list(self.dense_model.embed(query_texts))
        
        # search
        dense_scored_points_list = [
            self.manager.search(
                self.collection_name,
                query_dense_embedding,
                top_k
            )
            for query_dense_embedding in query_dense_embeddings
        ]
        
        # evaluate
        runs_dict = {}

        for i, query_id in enumerate(self.queries_df['_id'].values):
            runs_dict[query_id] = {}
            
            for scored_point in dense_scored_points_list[i]:
                doc_id = str(scored_point.payload['id'])
                runs_dict[query_id][doc_id] = float(scored_point.score)

        qrels_ranx = Qrels(self.qrels_dict)
        run_ranx = Run(runs_dict)

        return evaluate(qrels_ranx, run_ranx, metrics=metrics)