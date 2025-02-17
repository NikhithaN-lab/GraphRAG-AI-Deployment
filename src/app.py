import streamlit as st
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Neo4j connection details
NEO4J_URI = 'neo4j+s://32511ae0.databases.neo4j.io'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = 'xxxxxxx'# type your password!

# Initialize Neo4j connection
graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find similar reviews based on a query
def find_similar_reviews(query, top_n=5):
    # Generate query embedding
    try:
        query_embedding = model.encode(query)
    except Exception as e:
        st.error(f"Error generating query embedding: {e}")
        return []

    # Retrieve reviews from Neo4j
    try:
        query_result = graph.run("""
            MATCH (r:Review)
            WHERE r.embedding IS NOT NULL AND r.comment IS NOT NULL
            RETURN r.id AS review_id, r.embedding AS embedding, r.comment AS text
            LIMIT 1000
        """)
        results = list(query_result)
    except Exception as e:
        st.error(f"Error querying Neo4j: {e}")
        return []

    if not results:
        st.warning("No reviews found with embeddings.")
        return []

    # Compute similarities
    similarities = []
    for record in results:
        try:
            # Check if embedding is already a list or needs to be parsed
            embedding_data = record['embedding']
            if isinstance(embedding_data, str):
                # If embedding is stored as a string, try parsing it
                try:
                    review_embedding = np.array(json.loads(embedding_data))  # If stored as JSON string
                except json.JSONDecodeError:
                    review_embedding = np.array(eval(embedding_data))  # If stored as Python-style string
            elif isinstance(embedding_data, list):
                # If embedding is already a list, just use it as is
                review_embedding = np.array(embedding_data)
            else:
                # Handle unexpected cases where embedding is not a list or string
                st.warning(f"Unexpected embedding format for review ID {record['review_id']}")
                continue

            # Compute cosine similarity
            similarity = cosine_similarity([query_embedding], [review_embedding])
            similarities.append((record['review_id'], record['text'], similarity[0][0]))
        except Exception as e:
            st.warning(f"Error processing review ID {record['review_id']}: {e}")
            continue

    # Sort reviews by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)

    return similarities[:top_n]

# Streamlit Interface
st.title("Airbnb Review Chatbot")

user_query = st.text_input("Ask a question or describe a review:")

if user_query:
    st.write("Searching for similar reviews...")
    similar_reviews = find_similar_reviews(user_query, top_n=5)

    if similar_reviews:
        st.write("### Most Similar Reviews:")
        for i, review in enumerate(similar_reviews):
            # Format Review ID to show as an integer (whole number)
            formatted_review_id = int(review[0])  # Convert to integer to remove decimals
            # Display only Review ID and Review Text directly
            st.markdown(f"**{i+1}. Review ID:** {formatted_review_id}")
            st.markdown(f"{review[1]}")
    else:
        st.write("No similar reviews found. Try refining your query.")
