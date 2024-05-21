from neo4j import GraphDatabase
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

# Neo4j connection
neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def get_graph(graph_id):
    with neo4j_driver.session() as session:
        result = session.run("MATCH (g:Graph {id: $id})-[:RELATION*]->(n) RETURN g, n", id=graph_id)
        return result.data()

# PostgreSQL connection
conn = psycopg2.connect(database="embeddings_db", user="user", password="password", host="127.0.0.1", port="5432")
register_vector(conn)

def find_similar_embedding(query_vector):
    with conn.cursor() as cursor:
        cursor.execute("SELECT graph_id, embedding <-> %s AS distance FROM embeddings ORDER BY distance LIMIT 1;", (query_vector,))
        result = cursor.fetchone()
        return result[0]  # Return the graph_id

# Example usage
query_vector = np.array([0.1, 0.2, 0.3, ...])  # Your query embedding
graph_id = find_similar_embedding(query_vector)
graph_data = get_graph(graph_id)
print(graph_data)
