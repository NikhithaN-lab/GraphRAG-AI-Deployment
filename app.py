import streamlit as st
from sentence_transformers import SentenceTransformer
from py2neo import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Connect to Neo4j
NEO4J_URI = "neo4j+s://32511ae0.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "HYKino3fm8r87dIde7v4FUZl0WPNHwCsXjzS6dlM4xI"

graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit UI
st.title('Review Similarity Search')

# User input
query = st.text_area("Enter a review query:")

# Function to retrieve similar reviews from Neo4j
def find_similar_reviews(query, top_n=5):
    # Get the embedding for the query
    query_embedding = model.encode(query)

    # Retrieve all review embeddings from Neo4j
    query_result = graph.run("MATCH (r:Review) WHERE EXISTS(r.embedding) RETURN r.id, r.embedding LIMIT 1000")
    
    # List to store similarity scores
    similarities = []
    
    for record in query_result:
        review_embedding = np.array(record['r.embedding'])
        similarity = cosine_similarity([query_embedding], [review_embedding])
        similarities.append((record['r.id'], similarity[0][0]))
    
    # Sort by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return the top N most similar reviews
    return similarities[:top_n]

# Button to start the similarity search
if st.button('Search Similar Reviews'):
    if query:
        similar_reviews = find_similar_reviews(query)
        st.write("Top Similar Reviews:")
        for review_id, similarity in similar_reviews:
            st.write(f"Review ID: {review_id}, Similarity: {similarity:.4f}")
    else:
        st.write("Please enter a review query to search.")
