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

def create_graph(graph_id, nodes):
    with neo4j_driver.session() as session:
        session.run("CREATE (g:Graph {id: $id})", id=graph_id)
        for node in nodes:
            session.run("MATCH (g:Graph {id: $graph_id}) CREATE (n:Node {id: $node_id}), (g)-[:RELATION]->(n)",
                        graph_id=graph_id, node_id=node['id'])

# PostgreSQL connection
conn = psycopg2.connect(database="embeddings_db", user="user", password="password", host="127.0.0.1", port="5432")
register_vector(conn)

def insert_embedding(graph_id, embedding):
    with conn.cursor() as cursor:
        cursor.execute("INSERT INTO embeddings (graph_id, embedding) VALUES (%s, %s)", (graph_id, embedding))
        conn.commit()

def find_similar_embedding(query_vector):
    with conn.cursor() as cursor:
        cursor.execute("SELECT graph_id, embedding <-> %s AS distance FROM embeddings ORDER BY distance LIMIT 1;", (query_vector,))
        result = cursor.fetchone()
        return result[0]  # Return the graph_id

# Example usage
graph_id = 'graph_123'
nodes = [{'id': 'node_1'}, {'id': 'node_2'}]  # Example nodes
create_graph(graph_id, nodes)

embedding = np.random.rand(300)  # Example 300-dimensional embedding
insert_embedding(graph_id, embedding)

query_vector = np.random.rand(300)  # Your query embedding
similar_graph_id = find_similar_embedding(query_vector)
graph_data = get_graph(similar_graph_id)
print(graph_data)
