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

# Function to find similar reviews based on a query
def find_similar_reviews(query, top_n=5):
    # Encode the query using the SentenceTransformer model
    query_embedding = model.encode(query)

    # Query Neo4j for reviews with embeddings
    query_result = graph.run(
        """
        MATCH (r:Review)
        WHERE r.embedding IS NOT NULL AND r.review_text IS NOT NULL
        RETURN r.id AS id, r.embedding AS embedding, r.review_text AS text
        LIMIT 1000
        """
    )
    results = list(query_result)

    if not results:
        return []  # Return an empty list if no results found

    similarities = []

    # Calculate similarities
    for record in results:
        try:
            review_embedding = np.array(eval(record['embedding']))  # Convert string to list
            similarity = cosine_similarity([query_embedding], [review_embedding])
            similarities.append((record['id'], record['text'], similarity[0][0]))
        except Exception:
            continue  # Skip if there's an error processing a record

    # Sort results by similarity score in descending order
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Return the top N results
    return similarities[:top_n]

# Streamlit user interface
st.title("Airbnb Review Similarity Search")
query = st.text_area("Enter a review or query to find similar reviews:")

# If a query is entered, find similar reviews
if st.button("Find Similar Reviews"):
    if query.strip():
        similar_reviews = find_similar_reviews(query)
        if similar_reviews:
            st.write("Top similar reviews:")
            for review_id, review_text, similarity_score in similar_reviews:
                st.write(f"**Review ID:** {review_id}")
                st.write(f"**Similarity Score:** {similarity_score:.2f}")
                st.write(f"**Review Text:** {review_text}")
                st.write("---")
        else:
            st.write("No similar reviews found. Try refining your query.")
    else:
        st.write("Please enter a valid review or query.")
