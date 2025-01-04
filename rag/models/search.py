from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, 
    ScoredPoint, 
    Prefetch,
    SearchParams,
    Vector, 
    SparseVector
)

from .config import (
    DenseModelConfig,
    SparseModelConfig,
    RerankingModelConfig
)
from .metadata import Metadata


class HybridSearch(BaseModel):
    qdrant_client: QdrantClient
    dense_model_config: DenseModelConfig
    sparse_model_config: SparseModelConfig
    
    def create_collection(self, collection_name: str):
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                self.dense_model_config.name: self.dense_model_config.vector_params
            },
            sparse_vectors_config={
                self.sparse_model_config.name: self.sparse_model_config.sparse_vector_params
            }
        )
    
    def delete_collection(self, collection_name: str):
        self.qdrant_client.delete_collection(collection_name)
        
    def upsert(
        self, 
        collection_name: str, 
        dense_embeddings: list[Vector], 
        sparse_embeddings: list[SparseVector], 
        metadatas: list[Metadata]
    ) -> str:
        items = zip(dense_embeddings, sparse_embeddings, metadatas)
        
        result = self.qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=metadata.id, 
                    vector={
                        self.dense_model_config.name: dense_embedding,
                        self.sparse_model_config.name: sparse_embedding
                    },
                    payload={
                        'text': metadata.text
                    }
                )
                for dense_embedding, sparse_embedding, metadata in items
            ]
        )
        
        return str(result.status)
    
    def search(
        self, 
        collection_name: str, 
        dense_embedding: Vector, 
        sparse_embedding: SparseVector, 
        limit: int
    ) -> list[ScoredPoint]:
        return self.qdrant_client.search(
            collection_name=collection_name,
            query_vector={
                self.dense_model_config.name: dense_embedding,
                self.sparse_model_config.name: sparse_embedding
            },
            params=SearchParams(
                hnsw_ef=128
            ),
            limit=limit,
            with_payload=True
        )
        

class HybridRerankingSearch(BaseModel):
    qdrant_client: QdrantClient
    dense_model_config: DenseModelConfig
    sparse_model_config: SparseModelConfig
    reranking_model_config: RerankingModelConfig
    
    def create_collection(self, collection_name: str):
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                self.dense_model_config.name: self.dense_model_config.vector_params,
                self.reranking_model_config.name: self.reranking_model_config.vector_params
            },
            sparse_vectors_config={
                self.sparse_model_config.name: self.sparse_model_config.sparse_vector_params
            }
        )
    
    def delete_collection(self, collection_name: str):
        self.qdrant_client.delete_collection(collection_name)
        
    def upsert(
        self, 
        collection_name: str, 
        dense_embeddings: list[Vector],
        sparse_embeddings: list[SparseVector],
        reranking_embeddings: list[Vector],
        metadatas: list[Metadata]
    ) -> str:
        items = zip(dense_embeddings, sparse_embeddings, reranking_embeddings, metadatas)
        
        result = self.qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=metadata.id, 
                    vector={
                        self.dense_model_config.name: dense_embedding,
                        self.sparse_model_config.name: sparse_embedding,
                        self.reranking_model_config.name: reranking_embedding
                    },
                    payload={
                        'text': metadata.text
                    }
                )
                for dense_embedding, sparse_embedding, reranking_embedding, metadata in items
            ]
        )
        
        return str(result.status)
    
    def search(
        self, 
        collection_name: str, 
        dense_embedding: Vector, 
        sparse_embedding: SparseVector,
        reranking_embedding: Vector,
        prefetch_limit: int,
        limit: int
    ) -> list[ScoredPoint]:
        return self.qdrant_client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=dense_embedding,
                    using=self.dense_model_config.name,
                    params=SearchParams(
                        hnsw_ef=128
                    ),
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=sparse_embedding,
                    using=self.sparse_model_config.name,
                    limit=prefetch_limit,
                )
            ],
            query=reranking_embedding,
            using=self.reranking_model_config.name,
            with_payload=True,
            limit=limit,
        ).points