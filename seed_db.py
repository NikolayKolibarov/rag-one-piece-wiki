import glob

import numpy as np
from sentence_transformers import SentenceTransformer

from db import duckdb_connection


def seed_data():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    with duckdb_connection("one_piece_db.db") as con:
        con.execute('''
            CREATE TABLE IF NOT EXISTS one_piece_embeddings (
                text_content TEXT,
                embedding BLOB
            )
        ''')

        txt_files = glob.glob('data/*.txt')

        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as file:
                text_content = file.read()

            embedding = model.encode(text_content)
            embedding_blob = np.array(embedding).astype(np.float32).tobytes()

            con.execute("INSERT INTO one_piece_embeddings (text_content, embedding) VALUES (?, ?)",
                        (text_content, embedding_blob))


if __name__ == "__main__":
    seed_data()
