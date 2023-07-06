import math
import os
from typing import List, Optional

import numpy as np
import pinecone
import weaviate
from datasets import load_dataset
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    CollectionStatus,
    Distance,
    PointStruct,
    UpdateStatus,
    VectorParams,
)
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class VectorDatabase:
    """VectorDatabase class initializes the Vector Database index_name and loads the dataset
    for the usage of the subclasses."""

    def __init__(self, index_name, top_k: int = 3, select_range: Optional[int] = None):
        self.index_name = index_name
        logger.info(f"Index name: {self.index_name} initialized")
        if select_range:
            self.dataset = load_dataset(
                "Cohere/wikipedia-22-12-simple-embeddings", split="train"
            ).select(range(select_range))
        else:
            # Load the full dataset
            self.dataset = load_dataset(
                "Cohere/wikipedia-22-12-simple-embeddings", split="train"
            )
        logger.info(f"Dataset loaded with {len(self.dataset)} records")
        self.top_k = top_k
        self.dimension = 768

    def upsert(self) -> str:
        raise NotImplementedError

    def query(self, query_embedding: List[float]) -> dict:
        raise NotImplementedError


class PineconeDB(VectorDatabase):
    """PineconeDB class is a subclass of VectorDatabase that
    interacts with the Pinecone Cloud Vector Database index and
    has the following methods:
    - upsert: Upserts the embedding into the Pinecone index along with the metadata
    - query: Queries the Pinecone index with the query embedding along with the metadata
    - delete_index: Deletes the Pinecone index
    """

    def __init__(self, index_name):
        super().__init__(index_name)
        self.batch_size = 100  # Adjust the batch size as per your requirements
        pinecone.init(
            api_key=os.environ["PINECONE_API_KEY"],
            environment=os.environ["PINECONE_ENVIRONMENT"],
        )
        # Create the index if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=self.dimension, metric="cosine")

        # Connect to the index
        self.pinecone_index = pinecone.Index(index_name=index_name)

    def upsert(self) -> str:
        logger.info(f"pinecone batch size: {self.batch_size}")
        logger.info(f"total vectors from upsert: {len(self.dataset)}")
        num_vectors = len(self.dataset)
        logger.info(f"total num of vectors from upsert: {num_vectors}")
        num_batches = math.ceil(num_vectors / self.batch_size)

        logger.info(f"Upserting {num_vectors} vectors in {num_batches} batches")

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_vectors)

            vectors_batch = [
                (
                    f"{self.dataset[j]['id']}",
                    self.dataset[j]["emb"],
                    {"text": self.dataset[j]["text"]},
                )
                for j in range(start_idx, end_idx)
            ]

            logger.info(
                f"Upserting batch {i + 1} of {num_batches}, from {start_idx} to {end_idx} to Pinecone"
            )

            self.pinecone_index.upsert(vectors_batch)

        logger.info(f"Upserted {num_vectors} vectors to Pinecone")

        return "Upserted successfully"

    def query(self, query_embedding: List[float]) -> dict:
        # Pinecone Output:
        # {
        #     "matches": [
        #         {
        #             "id": "9",
        #             "metadata": {
        #                 "text": "In the Old Testament, Almighty God is the one who created the world. The God of the Old Testament is not always presented as the only God who exists Even though there may be other gods, the God of the Old Testament is always shown as the only God whom Israel is to worship. The God of the Old Testament is the one 'true God'; only Yahweh is Almighty. Both Jews and Christians have always interpreted the Bible (both the 'Old' and 'New' Testaments) as an affirmation of the oneness of Almighty God."
        #             },
        #             "score": 40.6401978,
        #             "values": [0.479291856, ..., 0.31344567],
        #         }
        #     ],
        #     "namespace": "",
        # }
        result = self.pinecone_index.query(
            vector=query_embedding,
            top_k=self.top_k,
            include_values=False,
            include_metadata=True,
        )
        return result.to_dict()

    def delete_index(self) -> str:
        pinecone.delete_index(self.index_name)
        logger.info(f"Pinecone Index Deleted: {self.index_name}")
        return "Pinecone Index deleted"


class QdrantDB(VectorDatabase):
    """QdrantDB class is a subclass of VectorDatabase that
    interacts with the Qdrant Cloud Vector Database. It has the following methods:
    - upsert: Upserts the dataset into the Qdrant collection(index) with the payload(metadata)
    - query: Queries the Qdrant collection(index) with the query embedding along with
    the payload(metadata)
    - delete_index: Deletes the Qdrant collection(index)
    """

    def __init__(self, index_name):
        super().__init__(index_name)
        self.batch_size = 1000  # Adjust the batch size as per your requirements

        self.qdrant_client = QdrantClient(
            os.environ["QDRANT_URL"],
            prefer_grpc=True,
            api_key=os.environ["QDRANT_API_KEY"],
        )

        qdrant_collections = self.qdrant_client.get_collections()
        # logger.info(f"qdrant collections: {qdrant_collections.collections}")

        # If no collections exist or if the index_name is not present in the collections, create the collection
        if len(qdrant_collections.collections) == 0 or not any(
            self.index_name in collection.name
            for collection in qdrant_collections.collections
        ):
            self.qdrant_client.recreate_collection(
                collection_name=self.index_name,
                vectors_config=VectorParams(
                    size=self.dimension, distance=Distance.COSINE
                ),
            )

            collection_info = self.qdrant_client.get_collection(
                collection_name=self.index_name
            )
            if collection_info.status == CollectionStatus.GREEN:
                logger.info(
                    f"Collection {self.index_name} created successfully in Qdrant"
                )

    def upsert(self) -> str:
        logger.info(f"total vectors from upsert: {len(self.dataset)}")
        num_vectors = len(self.dataset)
        logger.info(f"total num of vectors from upsert: {num_vectors}")
        num_batches = math.ceil(num_vectors / self.batch_size)
        logger.info(f"Qdrant batch size: {self.batch_size}")

        logger.info(f"Upserting {num_vectors} vectors in {num_batches} batches")

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_vectors)

            vectors_batch = [
                PointStruct(
                    id=self.dataset[j]["id"],
                    vector=self.dataset[j]["emb"],
                    payload={"text": self.dataset[j]["text"]},
                )
                for j in range(start_idx, end_idx)
            ]

            logger.info(
                f"Upserting batch {i + 1} of {num_batches}, from {start_idx} to {end_idx} to Qdrant"
            )

            operation_info = self.qdrant_client.upsert(
                collection_name=self.index_name, wait=True, points=vectors_batch
            )

            if operation_info.status != UpdateStatus.COMPLETED:
                raise Exception("Upsert failed")

        logger.info(f"Upserted {num_vectors} vectors to Qdrant")

        return "Qdrant Upserted successfully"

    def query(self, query_embedding: List[float]) -> dict:
        # Qdrant Output:
        # [
        #     ScoredPoint(
        #         id=6,
        #         version=0,
        #         score=0.8764744997024536,
        #         payload={
        #             "text": 'Tertullian was probably the first person to call these books the "Old Testament." He used the Latin name "vetus testamentum" in the 2nd century.'
        #         },
        #         vector=None,
        #     ),
        #     ScoredPoint(
        #         id=11,
        #         version=0,
        #         score=0.8760372996330261,
        #         payload={
        #             "text": "Other themes in the Old Testament include salvation, redemption, divine judgment, obedience and disobedience, faith and faithfulness. Throughout there is a strong emphasis on ethics and ritual purity. God demands both."
        #         },
        #         vector=None,
        #     ),
        #     ScoredPoint(
        #         id=592,
        #         version=5,
        #         score=0.8568843007087708,
        #         payload={
        #             "text": '"We stand here today as nothing more than a representative of the millions of our people who dared to rise up against a social operation whose very essence is war, violence, racism, oppression, repression and the impoverishment of an entire people."'
        #         },
        #         vector=None,
        #     ),
        # ]
        result = self.qdrant_client.search(
            collection_name=self.index_name,
            query_vector=query_embedding,
            limit=self.top_k,
            with_payload=True,
        )
        logger.info(f"Qdrant query result: {result}, type: {type(result)}")
        result_dict = []
        for point in result:
            point_dict = {
                "id": point.id,
                "version": point.version,
                "score": point.score,
                "payload": point.payload,
                "vector": point.vector,
            }
            result_dict.append(point_dict)
        return result_dict

    def delete_index(self) -> str:
        self.qdrant_client.delete_collection(collection_name=self.index_name)
        logger.info(f"Qdrant Collection/Index deleted: {self.index_name}")
        return "Qdrant Collection/Index deleted"


class WeaviateDB(VectorDatabase):
    def __init__(self, index_name):
        super().__init__(index_name)
        self.batch_size = 1000

        self.weaviate_class = "WikipediaEmbeddings"

        # Instantiate the client with the auth config
        self.weaviate_client = weaviate.Client(
            url=os.environ["WEAVIATE_URL"],  # Replace w/ your endpoint
            auth_client_secret=weaviate.auth.AuthApiKey(
                api_key=os.environ["WEAVIATE_API_KEY"]
            ),  # Replace w/ your API Key for the Weaviate instance
        )
        logger.info(f"weaviate schema: {self.weaviate_client.schema.get()}")

        schema = self.weaviate_client.schema.get()
        if any(d["class"] == self.weaviate_class for d in schema["classes"]):
            logger.info(f"schema for {self.weaviate_class} already exists")
        else:
            logger.info(f"schema for {self.weaviate_class} does not exist, creating it")
            schema = {
                "classes": [
                    {
                        "class": self.weaviate_class,
                        "description": "Contains the paragraph of text from Simple Wikipedia along with their embeddings",
                        "vectorizer": "none",
                        "properties": [
                            {
                                "name": "text",
                                "dataType": ["text"],
                            }
                        ],
                    }
                ]
            }

            self.weaviate_client.schema.create(schema)

    def upsert(self) -> str:
        logger.info(f"Weaviate batch size: {self.batch_size}")
        self.weaviate_client.batch.configure(batch_size=self.batch_size)

        with self.weaviate_client.batch as batch:
            for data in self.dataset:
                # id = data['id']
                text = data["text"]
                ebd = data["emb"]
                batch_data = {"text": text}
                batch.add_data_object(
                    data_object=batch_data, class_name=self.weaviate_class, vector=ebd
                )

        logger.info("All data added to weaviate")

        return "Upserted successfully"

    def query(self, query_embedding: List[float]) -> dict:
        # Weaviate Output:
        # {
        #     "result": {
        #         "data": {
        #             "Get": {
        #                 "WikipediaEmbeddings": [
        #                     {
        #                         "_additional": {"certainty": 0.9381886720657349},
        #                         "text": 'Tertullian was probably the first person to call these books the "Old Testament." He used the Latin name "vetus testamentum" in the 2nd century.',
        #                     },
        #                     {
        #                         "_additional": {"certainty": 0.9381886720657349},
        #                         "text": 'Tertullian was probably the first person to call these books the "Old Testament." He used the Latin name "vetus testamentum" in the 2nd century.',
        #                     },
        #                     {
        #                         "_additional": {"certainty": 0.9374131262302399},
        #                         "text": "Other themes in the Old Testament include salvation, redemption, divine judgment, obedience and disobedience, faith and faithfulness. Throughout there is a strong emphasis on ethics and ritual purity. God demands both.",
        #                     },
        #                 ]
        #             }
        #         }
        #     }
        # }
        vec = {"vector": query_embedding}
        result = (
            self.weaviate_client.query.get(self.weaviate_class, ["text"])
            .with_near_vector(vec)
            .with_limit(self.top_k)
            .with_additional(["certainty"])
            .do()
        )
        # logger.info(f"weaviate result: {result}")
        return result

    def delete_index(self) -> str:
        self.weaviate_client.schema.delete_class(self.weaviate_class)
        logger.info(f"Weaviate Class/Index deleted: {self.weaviate_class}")
        return "Weaviate Index/Class deleted"


class RedisDB(VectorDatabase):
    def __init__(self, index_name):
        super().__init__(index_name)
        self.batch_size = 1000  # Adjust the batch size as per your requirements
        # self.redis_client = redis.from_url(url="redis://localhost:6379")
        self.redis_client = redis.from_url(url=os.environ["REDIS_URL"])

        logger.info("Redis client initialized")

        def check_redis_index() -> bool:
            """Check if Redis index exists."""
            logger.info(f"Checking if index {self.index_name} exists")
            try:
                self.redis_client.ft(self.index_name).info()
            except:  # noqa: E722
                logger.info(f"Index {self.index_name} does not exist")
                return False
            logger.info(f"Index {self.index_name} already exists")
            return True

        if not check_redis_index():
            logger.info(f"Creating index {self.index_name}")
            # Constants
            schema = (
                TextField(name="text"),
                TextField(name="metadata"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.dimension,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            prefix = f"doc:{self.index_name}"

            # Create Redis Index
            self.redis_client.ft(index_name=self.index_name).create_index(
                fields=schema,
                definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
            )

    def upsert(self) -> str:
        # Write data to redis
        pipeline = self.redis_client.pipeline(transaction=False)
        redis_batch_count = 1
        for i, data in enumerate(self.dataset):
            # Use provided values by default or fallback
            key = data["id"]
            metadata = data["title"]
            embedding = data["emb"]
            text = data["text"]
            pipeline.hset(
                key,
                mapping={
                    "text": text,
                    "vector": np.array(embedding, dtype=np.float32).tobytes(),
                    "metadata": metadata,
                },
            )

            # Write batch
            if i % self.batch_size == 0:
                pipeline.execute()
                logger.info(f"Redis batch {redis_batch_count} written")
                redis_batch_count += 1

        # Cleanup final batch
        pipeline.execute()

        return "Redis Upserted successfully"

    def query(self, query_embedding: List[float]) -> dict:
        # base_query = (
        #     f"(@metadata:{ Old Testament })=>[KNN {self.top_k} @vector $query_embedding AS score]"
        # )
        # logger.info(f"base query: {base_query}")
        query = (
            Query("*=>[KNN 3 @vector $query_embedding AS score]")
            .return_fields("text", "score")
            .dialect(2)
        )

        query_params = {
            "query_embedding": np.array(query_embedding, dtype=np.float32).tobytes()
        }
        # logger.info(f"byte embedding: {query_params['query_embedding']}")
        result = self.redis_client.ft(self.index_name).search(query, query_params)
        logger.info(f"result: {result}, type: {type(result)}")
        return result

    def delete_index(self) -> str:
        self.redis_client.ft(self.index_name).drop_index(delete_documents=True)
        logger.info(f"Redis Index deleted: {self.index_name}")
        return "Redis Index deleted"
