# similarity-search
import streamlit as st
from sentence_transformers import SentenceTransformer
from py2neo import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize model and Neo4j connection
model = SentenceTransformer('all-MiniLM-L6-v2')
graph = Graph("neo4j+s://your-neo4j-uri", auth=("neo4j", "your-password"))

# Function to retrieve stored embeddings from Neo4j
def get_embeddings_from_neo4j():
    query = "MATCH (r:Review) WHERE EXISTS(r.embedding) RETURN r.id, r.embedding"
    result = graph.run(query)
    return result.data()

# Function for similarity search
def find_similar_reviews(query_text, top_n=5):
    # Convert the query text to an embedding
    query_embedding = model.encode(query_text).reshape(1, -1)
    
    # Fetch stored embeddings from Neo4j
    stored_data = get_embeddings_from_neo4j()
    stored_embeddings = [np.array(item['r.embedding']) for item in stored_data]
    stored_ids = [item['r.id'] for item in stored_data]
    
    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, stored_embeddings)
    
    # Get top N most similar reviews
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    top_reviews = [(stored_ids[i], similarities[0][i]) for i in top_indices]
    return top_reviews

# Streamlit UI
st.title("Review Similarity Search")
query = st.text_area("Enter a review or query:")

if query:
    # Perform similarity search
    similar_reviews = find_similar_reviews(query)
    st.write("Top Similar Reviews:")
    for review in similar_reviews:
        st.write(f"Review ID: {review[0]}, Similarity Score: {review[1]:.4f}")
