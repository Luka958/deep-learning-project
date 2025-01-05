from fastembed import (
    TextEmbedding, 
    SparseTextEmbedding, 
    LateInteractionTextEmbedding
)
from pandas import DataFrame
from qdrant_client.models import Fusion
from ranx import evaluate
from rag.base import BaseRepository
from rag.utils import get_qrels, get_run

from .metadata import Metadata


class Evaluator:
    def __init__(
        self,
        collection_name: str,
        dfs: tuple[DataFrame, DataFrame, DataFrame],
        repository: BaseRepository,
        dense_model: TextEmbedding | None = None,
        sparse_model: SparseTextEmbedding | None = None,
        reranking_model: LateInteractionTextEmbedding | None = None
        
    ):
        self.repository = repository
        self.dense_model = dense_model
        self.sparse_model = sparse_model
        self.reranking_model = reranking_model
        self.collection_name = collection_name
        self.corpus_df, self.queries_df, self.qrels_df = dfs
        
        self.qrels = get_qrels(self.qrels_df)
    
    def setup(self):
        # metadata
        corpus_texts: list[str] = self.corpus_df['text'].values.tolist()
        metadatas = [
            Metadata(
                id=row['_id'],
                text=row['text']
            )
            for _, row in self.corpus_df.iterrows()
        ]

        # embed        
        dense_embeddings = list(self.dense_model.embed(corpus_texts)) if self.dense_model else None
        sparse_embeddings = list(self.sparse_model.embed(corpus_texts)) if self.sparse_model else None
        reranking_embeddings = list(self.reranking_model.embed(corpus_texts)) if self.reranking_model else None
        
        # index
        self.repository.create_collection(self.collection_name)
        
        self.repository.upload_points(
            self.collection_name,
            metadatas,
            dense_embeddings,
            sparse_embeddings,
            reranking_embeddings
        )
    
    def run(
        self,
        metrics: list[str],
        top_k: int,
        scale_k: int | None = None,
        fusion_algorithm: Fusion | None = None
    ) -> dict[str, float] | float:
        # embed
        query_texts: list[str] = self.queries_df['text'].values.tolist()
        
        query_dense_embeddings = list(self.dense_model.embed(query_texts)) if self.dense_model else None
        query_sparse_embeddings = list(self.sparse_model.embed(query_texts)) if self.sparse_model else None
        query_reranking_embeddings = list(self.reranking_model.embed(query_texts)) if self.reranking_model else None
                
        # search
        scored_points_list = [
            self.repository.search(
                self.collection_name,
                top_k,
                int(top_k * scale_k) if scale_k else None,
                fusion_algorithm,
                query_dense_embedding,
                query_sparse_embeddings,
                query_reranking_embeddings
            )
            for query_dense_embedding in query_dense_embeddings
        ]
        
        # evaluate
        run = get_run(self.queries_df, scored_points_list)

        return evaluate(self.qrels, run, metrics=metrics)
    
    def clear(self) -> bool:
        return self.repository.delete_collection(self.collection_name)