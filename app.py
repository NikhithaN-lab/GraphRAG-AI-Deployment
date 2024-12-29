import streamlit as st
from py2neo import Graph
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json  # Added for debugging JSON formatting if necessary

# Neo4j connection details
NEO4J_URI = 'neo4j+s://32511ae0.databases.neo4j.io'
NEO4J_USERNAME = 'neo4j'
NEO4J_PASSWORD = 'HYKino3fm8r87dIde7v4FUZl0WPNHwCsXjzS6dlM4xI'

# Initialize Neo4j connection
graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to find similar reviews based on a query
def find_similar_reviews(query, top_n=5):
    st.write("Processing your query...")  # Debugging step
    
    # Generate query embedding
    try:
        query_embedding = model.encode(query)
        st.write("Query embedding generated successfully.")
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
        st.write(f"Retrieved {len(results)} reviews with embeddings.")  # Debugging step to show count of results
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
            # Remove the part where embeddings are printed out
            # st.write(f"Embedding Example: {record['embedding']}")  # Removed this line
            # st.write(f"Raw Query Results: {results[:2]}")  # Removed this line

            # Ensure embedding is properly converted (from JSON-style or Python string to array)
            embedding_str = record['embedding']
            try:
                review_embedding = np.array(json.loads(embedding_str))  # If stored as JSON string
            except json.JSONDecodeError:
                review_embedding = np.array(eval(embedding_str))  # If stored as Python-style string

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
            st.write(f"**{i+1}. Review ID:** {review[0]}")
            st.write(f"**Similarity Score:** {review[2]:.4f}")
            st.write(f"**Review Text:** {review[1]}")
    else:
        st.write("No similar reviews found. Try refining your query.")

# Debugging: Add additional Neo4j inspection code if necessary
# If results aren't returned correctly, you can inspect the raw embeddings in Neo4j like this:
st.write("Debugging raw data from Neo4j...")
try:
    raw_data = graph.run("""
        MATCH (r:Review)
        WHERE r.embedding IS NOT NULL
        RETURN r.id, r.embedding
        LIMIT 5
    """)
    st.write(list(raw_data))  # Display raw data for inspection
except Exception as e:
    st.error(f"Error querying Neo4j for raw data: {e}")
