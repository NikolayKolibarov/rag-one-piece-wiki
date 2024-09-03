import duckdb
from contextlib import contextmanager


@contextmanager
def duckdb_connection(database_path=None):
    if database_path:
        con = duckdb.connect(database_path)
    else:
        con = duckdb.connect()
    try:
        yield con
    finally:
        con.close()


def get_all_documents():
    with duckdb_connection("one_piece_db.db") as con:
        results = con.execute("SELECT text_content, embedding FROM one_piece_embeddings").fetchall()

    return results
