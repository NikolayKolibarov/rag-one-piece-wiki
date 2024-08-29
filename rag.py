import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer


def get_embeddings(model, text):
    # Generate embedding for the text data
    return model.encode(text)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Query embeddings and perform similarity search
def search_similar_texts(con, query_embedding, top_k=3):
    # Fetch all embeddings from DuckDB
    results = con.execute("SELECT text_content, embedding FROM one_piece_embeddings").fetchall()

    similarities = []
    for row in results:
        text_content, embedding_blob = row
        stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        similarity = cosine_similarity(query_embedding, stored_embedding)
        similarities.append((text_content, similarity))

    # Sort by similarity and get top K results
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def get_input(query_text):
    with open('doflamingo.txt', 'r', encoding='utf-8') as file:
        text_content = file.read()

    con = duckdb.connect()  # ':memory:' or no argument creates an in-memory database

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embedding for the query
    query_embedding = get_embeddings(model, query_text)

    # Create table for embeddings
    con.execute('''
        CREATE TABLE IF NOT EXISTS one_piece_embeddings (
            text_content TEXT,
            embedding BLOB
        )
    ''')


    embedding = get_embeddings(model, text_content)
    embedding_blob = np.array(embedding).astype(np.float32).tobytes()
    con.execute("INSERT INTO one_piece_embeddings (text_content, embedding) VALUES (?, ?)",
                (text_content, embedding_blob))
    con.close()

    # Get the text data from the table
    # texts = con.execute("SELECT text_content FROM one_piece_embeddings").fetchall()

    # Perform search
    search_results = search_similar_texts(con, query_embedding)
    search_results = [search_result for search_result in search_results if search_result[1] > 0.5]
    for result in search_results:
        print(f"Text: {result[0][:10]}, Similarity: {result[1]}")

    top_texts = [result[0] for result in search_results]
    context = "\n\n".join(top_texts)

    # Combine the context with the initial question
    if len(search_results) > 0:
        input_text = f"Based on the following information, answer the question:\n\n{context}\n\nQuestion: {query_text}"
    else:
        input_text = query_text

    return input_text

