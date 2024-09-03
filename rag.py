import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from db import get_all_documents


def get_embeddings(model, text):
    return model.encode(text)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def search_similar_texts(query_embedding, top_k=3):
    results = get_all_documents()

    similarities = []
    for row in results:
        text_content, embedding_blob = row
        stored_embedding = np.frombuffer(embedding_blob, dtype=np.float32)

        similarity = cosine_similarity(query_embedding, stored_embedding)
        similarities.append((text_content, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def get_similar_documents_by_text(query_text):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_embedding = get_embeddings(model, query_text)

    search_results = search_similar_texts(query_embedding)

    return search_results


def prepare_input(query_text):
    search_results = get_similar_documents_by_text(query_text)
    for result in search_results:
        print(f"Text: {result[0][:10]}, Similarity: {result[1]}")

    print("Filtered")
    search_results = [search_result for search_result in search_results if search_result[1] > 0.4]
    for result in search_results:
        print(f"Text: {result[0][:10]}, Similarity: {result[1]}")

    top_texts = [result[0] for result in search_results]
    context = "\n\n".join(top_texts)

    if len(search_results) > 0:
        input_text = f"Here is extra information for context:\n{context}\nQuestion: {query_text}"
    else:
        input_text = query_text

    return input_text


async def stream_ollama_response(user_message: str):
    model_input = prepare_input(user_message)

    print(model_input)

    stream = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': model_input}],
        stream=True,
    )

    for chunk in stream:
        yield chunk['message']['content']
