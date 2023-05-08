import os

import cohere
import pinecone
from datasets import load_dataset
from typing import List


class VectorDB:
    def __init__(self, index_name, top_k: int = 3):
        self.index_name = index_name
        # Load the dataset
        self.dataset = load_dataset(
            "Cohere/wikipedia-22-12-simple-embeddings", split="train"
        )
        self.top_k = 3

    def upsert(self) -> str:
        raise NotImplementedError

    def query(self, query_embedding: List[float]) -> dict:
        raise NotImplementedError


class PineconeDB(VectorDB):
    def __init__(self, index_name):
        super().__init__(index_name)
        self.batch_size = 50
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"],
        )
        # Create the index if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=768, metric="cosine")

        # Connect to the index
        self.index = pinecone.Index(index_name=index_name)

    def upsert(self) -> str:
        self.vectors = [
            (
                f"{self.dataset[i]['id']}",
                self.dataset[i]["emb"],
                {"text": self.dataset[i]["text"]},
            )
            for i in range(len(self.dataset))
        ]

        # Note: When upserting larger amounts of data, upsert data in batches
        # of 100 vectors or fewer over multiple upsert requests.

        # Upsert the vectors in batches of 50
        num_vectors = len(self.vectors)

        for i in range(0, num_vectors, self.batch_size):
            batch = self.vectors[i : i + self.batch_size]
            self.index.upsert(batch)

        return "Upserted successfully"

    def query(self, query_embedding: List[float]) -> dict:
        return self.index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_values=True,
            include_metadata=True,
        )


from fastapi import FastAPI, Body
from pydantic import BaseModel
import os
import cohere
from pinecone import PineconeDB

app = FastAPI()


# Define the request model
class QueryRequest(BaseModel):
    query: str


# Step 1: Create an index in Pinecone
index_name = "wikipedia-embeddings"

# Initialize Cohere
co = cohere.Client(os.environ["COHERE_API_KEY"])

# Define the index name
index_name = "wikipedia-embeddings"


# Dependency function to choose a vector database implementation
def get_vector_db() -> Type[VectorDatabase]:
    # Choose either PineconeDatabase or QdrantDatabase here
    vector_db_class = PineconeDB  # or QdrantDatabase
    return vector_db_class(index_name)


@app.on_event("startup")
async def startup_event():
    vector_db = get_vector_db()
    vector_db.upsert()


@app.post("/ask")
async def ask(
    request: QueryRequest, vector_db: VectorDatabase = Depends(get_vector_db)
):
    # Get the embeddings
    query_embeds = co.embed([request.query], model="multilingual-22-12").embeddings

    # Query the VectorDatabase
    result = vector_db.query(query_embedding=query_embeds[0])

    return {"result": result}


@app.on_event("shutdown")
async def shutdown():
    vector_db = get_vector_db()
    vector_db.delete_index()
