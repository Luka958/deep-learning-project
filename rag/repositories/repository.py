from fastembed.sparse import SparseEmbedding
from numpy import ndarray
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct, 
    ScoredPoint, 
    Prefetch,
    SearchParams,
    Fusion,
    FusionQuery
)
from qdrant_client.http.models import SparseVector, NamedVector, NamedSparseVector
from pydantic import BaseModel
from uuid import uuid4

from rag.models import (
    DenseModelConfig,
    SparseModelConfig,
    RerankingModelConfig,
    Metadata
)
from rag.base import BaseRepository


class DenseSearchRepository(BaseModel, BaseRepository):
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
    
    def delete_collection(self, collection_name: str) -> bool:
        return self.qdrant_client.delete_collection(collection_name)
        
    def upload_points(
        self, 
        collection_name: str, 
        metadatas: list[Metadata],
        dense_embeddings: list[ndarray] = None,
        sparse_embeddings: list[SparseEmbedding] = None,
        reranking_embeddings: list[ndarray] = None
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
        limit: int,
        prefetch_limit: int = None,
        fusion_algorithm: Fusion = None,
        dense_embedding: ndarray = None, 
        sparse_embedding: SparseEmbedding = None,
        reranking_embedding: ndarray = None
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


class SparseSearchRepository(BaseModel, BaseRepository):
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
    
    def delete_collection(self, collection_name: str) -> bool:
        return self.qdrant_client.delete_collection(collection_name)
        
    def upload_points(
        self, 
        collection_name: str, 
        metadatas: list[Metadata],
        dense_embeddings: list[ndarray] = None,
        sparse_embeddings: list[SparseEmbedding] = None,
        reranking_embeddings: list[ndarray] = None
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
        limit: int,
        prefetch_limit: int = None,
        fusion_algorithm: Fusion = None,
        dense_embedding: ndarray = None, 
        sparse_embedding: SparseEmbedding = None,
        reranking_embedding: ndarray = None
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


class HybridFusionSearchRepository(BaseModel, BaseRepository):
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
    
    def delete_collection(self, collection_name: str) -> bool:
        return self.qdrant_client.delete_collection(collection_name)
        
    def upload_points(
        self, 
        collection_name: str, 
        metadatas: list[Metadata],
        dense_embeddings: list[ndarray] = None,
        sparse_embeddings: list[SparseEmbedding] = None,
        reranking_embeddings: list[ndarray] = None
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
        limit: int,
        prefetch_limit: int = None,
        fusion_algorithm: Fusion = None,
        dense_embedding: ndarray = None, 
        sparse_embedding: SparseEmbedding = None,
        reranking_embedding: ndarray = None
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
        

class HybridRerankingSearchRepository(BaseModel, BaseRepository):
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
    
    def delete_collection(self, collection_name: str) -> bool:
        return self.qdrant_client.delete_collection(collection_name)
        
    def upload_points(
        self, 
        collection_name: str, 
        metadatas: list[Metadata],
        dense_embeddings: list[ndarray] = None,
        sparse_embeddings: list[SparseEmbedding] = None,
        reranking_embeddings: list[ndarray] = None
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
        limit: int,
        prefetch_limit: int = None,
        fusion_algorithm: Fusion = None,
        dense_embedding: ndarray = None, 
        sparse_embedding: SparseEmbedding = None,
        reranking_embedding: ndarray = None
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
