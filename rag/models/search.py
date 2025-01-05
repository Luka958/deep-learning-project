from pydantic import BaseModel
from fastembed.sparse import SparseEmbedding
from numpy import ndarray
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, 
    ScoredPoint, 
    Prefetch,
    SearchParams,
    Fusion,
    FusionQuery,
    OptimizersConfigDiff
)
from qdrant_client.http.models import SparseVector, NamedVector, NamedSparseVector
from uuid import uuid4

from .config import (
    DenseModelConfig,
    SparseModelConfig,
    RerankingModelConfig
)
from .metadata import Metadata


class DenseSearchManager(BaseModel):
    qdrant_client: QdrantClient
    dense_model_config: DenseModelConfig
    
    model_config = {'arbitrary_types_allowed': True}
    
    def create_collection(self, collection_name: str) -> bool: 
        return self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                self.dense_model_config.name: self.dense_model_config.vector_params
            }
        )
    
    def delete_collection(self, collection_name: str):
        self.qdrant_client.delete_collection(collection_name)
        
    def upload_points(
        self, 
        collection_name: str, 
        dense_embeddings: list[ndarray], 
        metadatas: list[Metadata]
    ):
        items = zip(dense_embeddings, metadatas)
        
        self.qdrant_client.upload_points(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=str(uuid4()), 
                    vector={
                        self.dense_model_config.name: dense_embedding,
                    },
                    payload={
                        'id': metadata.id,
                        'text': metadata.text
                    }
                )
                for dense_embedding, metadata in items
            ]
        )
    
    def search(
        self, 
        collection_name: str, 
        dense_embedding: ndarray, 
        limit: int
    ) -> list[ScoredPoint]:
        return self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=NamedVector(
                name=self.dense_model_config.name,
                vector=dense_embedding
            ),
            search_params=SearchParams(
                hnsw_ef=128
            ),
            limit=limit,
            with_payload=True
        )


class SparseSearchManager(BaseModel):
    qdrant_client: QdrantClient
    sparse_model_config: SparseModelConfig
    
    model_config = {'arbitrary_types_allowed': True}
    
    def create_collection(self, collection_name: str) -> bool:
        return self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={},
            sparse_vectors_config={
                self.sparse_model_config.name: self.sparse_model_config.sparse_vector_params
            }
        )
    
    def delete_collection(self, collection_name: str):
        self.qdrant_client.delete_collection(collection_name)
        
    def upload_points(
        self, 
        collection_name: str, 
        sparse_embeddings: list[SparseEmbedding], 
        metadatas: list[Metadata]
    ):
        items = zip(sparse_embeddings, metadatas)
        
        self.qdrant_client.upload_points(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=str(uuid4()), 
                    vector={
                        self.sparse_model_config.name: sparse_embedding.as_object()
                    },
                    payload={
                        'id': metadata.id,
                        'text': metadata.text
                    }
                )
                for sparse_embedding, metadata in items
            ]
        )
    
    def search(
        self, 
        collection_name: str, 
        sparse_embedding: SparseEmbedding, 
        limit: int
    ) -> list[ScoredPoint]:
        return self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=NamedSparseVector(
                name=self.sparse_model_config.name,
                vector=SparseVector(
                    indices=sparse_embedding.indices,
                    values=sparse_embedding.values
                )
            ),
            limit=limit,
            with_payload=True
        )


class HybridFusionSearchManager(BaseModel):
    qdrant_client: QdrantClient
    dense_model_config: DenseModelConfig
    sparse_model_config: SparseModelConfig
    
    model_config = {'arbitrary_types_allowed': True}
    
    def create_collection(self, collection_name: str) -> bool:
        return self.qdrant_client.create_collection(
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
        
    def upload_points(
        self, 
        collection_name: str, 
        dense_embeddings: list[ndarray], 
        sparse_embeddings: list[SparseEmbedding], 
        metadatas: list[Metadata]
    ):
        items = zip(dense_embeddings, sparse_embeddings, metadatas)
        
        self.qdrant_client.upload_points(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=str(uuid4()), 
                    vector={
                        self.dense_model_config.name: dense_embedding,
                        self.sparse_model_config.name: sparse_embedding.as_object()
                    },
                    payload={
                        'id': metadata.id,
                        'text': metadata.text
                    }
                )
                for dense_embedding, sparse_embedding, metadata in items
            ]
        )
    
    def search(
        self, 
        collection_name: str, 
        dense_embedding: ndarray, 
        sparse_embedding: SparseEmbedding, 
        fusion_algorithm: Fusion,
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
                    limit=limit
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_embedding.indices, 
                        values=sparse_embedding.values
                    ),
                    using=self.sparse_model_config.name,
                    limit=limit
                )
            ],
            query=FusionQuery(fusion=fusion_algorithm),
            with_payload=True
        ).points
        

class HybridRerankingSearchManager(BaseModel):
    qdrant_client: QdrantClient
    dense_model_config: DenseModelConfig
    sparse_model_config: SparseModelConfig
    reranking_model_config: RerankingModelConfig
    
    model_config = {'arbitrary_types_allowed': True}
    
    def create_collection(self, collection_name: str) -> bool:
        return self.qdrant_client.create_collection(
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
        
    def upload_points(
        self, 
        collection_name: str, 
        dense_embeddings: list[ndarray],
        sparse_embeddings: list[SparseEmbedding],
        reranking_embeddings: list[ndarray],
        metadatas: list[Metadata]
    ):
        items = zip(dense_embeddings, sparse_embeddings, reranking_embeddings, metadatas)
        
        self.qdrant_client.upload_points(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=str(uuid4()), 
                    vector={
                        self.dense_model_config.name: dense_embedding,
                        self.sparse_model_config.name: sparse_embedding.as_object(),
                        self.reranking_model_config.name: reranking_embedding
                    },
                    payload={
                        'id': metadata.id,
                        'text': metadata.text
                    }
                )
                for dense_embedding, sparse_embedding, reranking_embedding, metadata in items
            ],
            batch_size=20,
            parallel=6
        )
    
    def search(
        self, 
        collection_name: str, 
        dense_embedding: ndarray, 
        sparse_embedding: SparseEmbedding,
        reranking_embedding: ndarray,
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
                    query=SparseVector(
                        indices=sparse_embedding.indices, 
                        values=sparse_embedding.values
                    ),
                    using=self.sparse_model_config.name,
                    limit=prefetch_limit,
                )
            ],
            query=reranking_embedding,
            using=self.reranking_model_config.name,
            with_payload=True,
            limit=limit,
        ).points
