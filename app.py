import streamlit as st
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Neo4j connection details
NEO4J_URI = 'neo4j+s://32511ae0.databases.neo4j.io'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = 'HYKino3fm8r87dIde7v4FUZl0WPNHwCsXjzS6dlM4xI'

# Initialize the Neo4j graph
graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_similar_reviews(query, top_n=5):
    # Encode the query using the SentenceTransformer model
    query_embedding = model.encode(query)

    # Query Neo4j for reviews with embeddings
    query_result = graph.run("MATCH (r:Review) WHERE r.embedding IS NOT NULL RETURN r.id, r.embedding LIMIT 1000")
    
    similarities = []

    # Loop through query result and compute similarity
    for record in query_result:
        if 'r.embedding' in record:
            review_embedding = np.array(record['r.embedding'])
            similarity = cosine_similarity([query_embedding], [review_embedding])
            similarities.append((record['r.id'], similarity[0][0]))

    # Debugging: Display the number of reviews retrieved
    st.write(f"Retrieved {len(similarities)} reviews with embeddings.")

    # Sort the similarities and return top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit user interface for input query
st.title("Airbnb Review Similarity Search")
query = st.text_input("Enter a review or query to find similar reviews:")

# If a query is entered, find similar reviews
if query:
    similar_reviews = find_similar_reviews(query)
    for review in similar_reviews:
        st.write(f"Review ID: {review[0]} - Similarity Score: {review[1]}")
