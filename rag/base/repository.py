from abc import ABC, abstractmethod
from fastembed import SparseEmbedding
from numpy import ndarray
from qdrant_client.models import Fusion
from qdrant_client.http.models import ScoredPoint
from rag.models import Metadata

class BaseRepository(ABC):
    @abstractmethod
    def create_collection(self, collection_name: str) -> bool: 
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        pass
    
    @abstractmethod
    def upload_points(
        self, 
        collection_name: str,
        metadatas: list[Metadata],
        dense_embeddings: list[ndarray] = None,
        sparse_embeddings: list[SparseEmbedding] = None,
        reranking_embeddings: list[ndarray] = None
    ):
        pass
    
    @abstractmethod
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
        pass