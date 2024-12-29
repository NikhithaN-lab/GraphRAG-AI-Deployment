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
    st.write("Processing your query...")  # Debugging: Confirm function call
    
    # Encode the query using the SentenceTransformer model
    try:
        query_embedding = model.encode(query)
        st.write("Query embedding generated successfully.")
    except Exception as e:
        st.error(f"Error generating query embedding: {e}")
        return []

    # Query Neo4j for reviews with embeddings only
    try:
        query_result = graph.run("""
            MATCH (r:Review)
            WHERE r.embedding IS NOT NULL AND r.review_text IS NOT NULL
            RETURN r.id AS review_id, r.embedding AS embedding, r.review_text AS text
            LIMIT 1000
        """)
        results = list(query_result)
        st.write(f"Retrieved {len(results)} reviews with embeddings.")  # Debugging: Display count
    except Exception as e:
        st.error(f"Error querying Neo4j: {e}")
        return []

    if not results:
        st.warning("No reviews found with embeddings.")
        return []

    similarities = []

    # Process the results and calculate similarity scores
    for record in results:
        try:
            # Convert embedding from string to array
            review_embedding = np.array(eval(record['embedding']))
            st.write(f"Processing review ID {record['review_id']}...")  # Debugging: Current review

            # Compute cosine similarity
            similarity = cosine_similarity([query_embedding], [review_embedding])
            st.write(f"Similarity with review ID {record['review_id']}: {similarity[0][0]}")  # Debugging: Similarity score
            similarities.append((record['review_id'], record['text'], similarity[0][0]))
        except Exception as e:
            st.warning(f"Error processing review ID {record['review_id']}: {e}")
            continue

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[2], reverse=True)

    # Return the top N similar reviews
    return similarities[:top_n]

# Chatbot Interface
st.title("Airbnb Review Chatbot")

# User input: conversational query
user_query = st.text_input("Ask a question or enter a review description:")

if user_query:
    st.write("Searching for similar reviews...")
    similar_reviews = find_similar_reviews(user_query, top_n=5)

    if similar_reviews:
        st.write("Here are the most similar reviews:")
        for i, review in enumerate(similar_reviews):
            st.write(f"### Similar Review {i + 1}")
            st.write(f"**Review ID:** {review[0]}")
            st.write(f"**Similarity Score:** {review[2]:.4f}")
            st.write(f"**Review Text:** {review[1]}")
    else:
        st.write("No similar reviews found. Try refining your query.")
