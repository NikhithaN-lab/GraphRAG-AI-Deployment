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
    st.write("Starting to find similar reviews...")  # Debugging: Check if the function is called
    
    # Encode the query using the SentenceTransformer model
    try:
        query_embedding = model.encode(query)
        st.write("Query embedding generated.")  # Debugging: Check if query embedding is generated
    except Exception as e:
        st.write(f"Error generating query embedding: {e}")
        return []

    # Query Neo4j for reviews with embeddings
    try:
        query_result = graph.run("MATCH (r:Review) WHERE r.embedding IS NOT NULL RETURN r.id, r.embedding LIMIT 1000")
        results = list(query_result)
        st.write(f"Query Result: {results[:5]}")  # Debugging: Check first 5 results
    except Exception as e:
        st.write(f"Error querying Neo4j: {e}")
        return []

    if not results:
        st.write("No reviews found with embeddings.")  # Debugging: Check if no results are returned
        return []

    similarities = []

    # Check if the results contain embeddings and process them
    for record in results:
        if 'r.embedding' in record:
            try:
                # Ensure the embedding is in the correct numeric format
                review_embedding = np.array(eval(record['r.embedding']))  # Convert string to list if necessary
                st.write(f"Processing review {record['r.id']}, embedding: {review_embedding[:5]}...")  # Debugging: Check the first few values of the embedding

                # Compute the cosine similarity between the query and the review embedding
                similarity = cosine_similarity([query_embedding], [review_embedding])
                similarities.append((record['r.id'], similarity[0][0]))

                # Debugging: Print the similarity score for each review
                st.write(f"Similarity with review {record['r.id']}: {similarity[0][0]}")

            except Exception as e:
                st.write(f"Error processing embedding for review {record['r.id']}: {e}")
                continue

    st.write(f"Retrieved {len(similarities)} reviews with embeddings.")  # Debugging: Check number of reviews processed

    # Sort the results by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top N similar reviews
    return similarities[:top_n]

# Streamlit user interface for input query
st.title("Airbnb Review Similarity Search")
query = st.text_input("Enter a review or query to find similar reviews:")

# If a query is entered, find similar reviews
if query:
    st.write("Finding similar reviews for the query...")  # Debugging: Confirm the query is received
    similar_reviews = find_similar_reviews(query)
    if similar_reviews:
        for review in similar_reviews:
            st.write(f"Review ID: {review[0]} - Similarity Score: {review[1]}")
    else:
        st.write("No similar reviews found.")
