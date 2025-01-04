from pydantic import BaseModel
from qdrant_client.models import VectorParams, SparseVectorParams


class DenseModelConfig(BaseModel):
    name: str
    vector_params: VectorParams


class RerankingModelConfig(BaseModel):
    name: str
    vector_params: VectorParams


class SparseModelConfig(BaseModel):
    name: str
    sparse_vector_params: SparseVectorParams
