import os
from typing import Type

import cohere
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from vectordb import PineconeDB, QdrantDB, VectorDatabase, WeaviateDB

from time import perf_counter

# Initializations
app = FastAPI()


# Define the request model
class QueryRequest(BaseModel):
    query: str


class WrapperQA:
    def __init__(self):
        self.vector_db = {}
        # Define the index name
        self.index_name = "wikipedia-embeddings"

        # Initialize Cohere
        self.co = cohere.Client(os.environ["COHERE_API_KEY"])

        # Initialize the VectorDatabases
        self.vector_db["weaviate"] = WeaviateDB(self.index_name)
        self.vector_db["pinecone"] = PineconeDB(self.index_name)
        self.vector_db["qdrant"] = QdrantDB(self.index_name)

    def get_vector_db(self, db_name: str):
        return self.vector_db[db_name]

    def upsert(self):
        for db in self.vector_db.values():
            db.upsert()
        return {"status": "ok"}

    def query(self, request: QueryRequest):
        # Get the embeddings from Cohere
        query_embeds = qa_model.co.embed(
            [request.query], model="multilingual-22-12"
        ).embeddings
        logger.info(f"Query Embeddings: {query_embeds}")
        result_dict = {}
        # Query the VectorDatabase
        for db_name, db in self.vector_db.items():
            result_dict[db_name] = {}
            t1_start = perf_counter()
            result_dict[db_name]["query_result"] = db.query(
                query_embedding=query_embeds[0]
            )
            t1_stop = perf_counter()
            result_dict[db_name]["query_time"] = t1_stop - t1_start
        return result_dict

    def delete_index(self):
        for db in self.vector_db.values():
            db.delete_index()
        return {"status": "ok"}


qa_model = WrapperQA()


@app.post("/ask")
async def ask(request: QueryRequest) -> JSONResponse:
    logger.info(f"Received request: {request}")

    result = qa_model.query(request)

    logger.info(f"Result: {result}")
    logger.info(f"Result type: {type(result)}")
    result_dict = {"result": result}
    return JSONResponse(content=result_dict)


@app.post("/upsert")
async def upsert():
    return qa_model.upsert()


@app.post("/delete")
async def delete():
    return qa_model.delete_index()


@app.get("/health")
async def health():
    return {"status": "ok"}
