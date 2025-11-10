from typing import List
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, "r") as f:
        content = f.read()
    return [chunk for chunk in content.split('\n\n')]

def embed_chunk(chunk: str, embedding_model) -> List[float]:
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()

def save_embeddings(chunks: List[str], embeddings: List[List[float]], chromadb_collection) -> None:
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chromadb_collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )

def retrieve(query: str, top_k: int, chromadb_collection, embedding_model) -> List[str]:
    query_embedding = embed_chunk(query, embedding_model=embedding_model)
    results = chromadb_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]

def rerank(query: str, retrieved_chunks: List[str], top_k: int, cross_encoder) -> List[str]:
    if not retrieved_chunks:
        print("[WARNING] retrieved_chunks is None!!!")
        return []
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]

if __name__ == "__main__":
   
    print("Loading Embedding Model...")
    embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    print('Loading Chromadb...')
    chromadb_client = chromadb.PersistentClient("./dataset/doc.db")
    chromadb_collection = chromadb_client.get_or_create_collection(name="default")
    print('Loading CrossEncoder...')
    cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

    print("Chunking...")
    chunks = split_into_chunks("./dataset/info.md")
    print("Embedding...")
    embeddings = [embed_chunk(chunk, embedding_model) for chunk in chunks]
    print(len(chunks))
    print(chunks[0])
    print(embeddings[0])

    # Save to Chromadb
    save_embeddings(chunks=chunks, embeddings=embeddings, chromadb_collection=chromadb_collection)

    query = """- 健康度持续下降:133次;2025-06-05 01:06:24.492~01:06:24.993;SOH 0.00~0.00
    - 健康度低于阈值:294次;2025-06-05 01:06:22.906~01:06:24.485;SOH 0.00~87.58|惯组X轴(264次)
    - 健康度濒临阈值:3次;2025-06-05 01:06:24.996~01:06:25.002;SOH 0.00~0.56
    """

