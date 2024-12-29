import numpy as np
import streamlit as st
from py2neo import Graph
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize the Neo4j Graph connection
graph = Graph("bolt://localhost:7687", auth=("neo4j", "password"))

# Function to find similar reviews
def find_similar_reviews(query, top_n=5):
    query_embedding = model.encode(query)
    
    # Query Neo4j for reviews with embeddings (embedding property must be present)
    query_result = graph.run("MATCH (r:Review) WHERE r.embedding IS NOT NULL RETURN r.id, r.embedding LIMIT 1000")
    similarities = []
    
    # Loop through each review and calculate similarity
    for record in query_result:
        if 'r.embedding' in record:
            try:
                # Ensure that the 'embedding' is a valid numpy array
                review_embedding = np.array(record['r.embedding'])
                if review_embedding.size > 0:
                    similarity = cosine_similarity([query_embedding], [review_embedding])
                    similarities.append((record['r.id'], similarity[0][0]))
            except Exception as e:
                # Handle any errors in processing the embedding (e.g., invalid data)
                st.write(f"Error processing review ID {record['r.id']}: {e}")
    
    # Debugging: Print out how many reviews were retrieved with valid embeddings
    st.write(f"Retrieved {len(similarities)} reviews with embeddings.")
    
    # Sort reviews based on similarity and return the top N
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Streamlit UI setup
st.title("Similar Reviews Finder")

# User input for the query
query = st.text_input("Enter a review to find similar ones:")

# Button to trigger similarity search
if st.button('Find Similar Reviews'):
    if query:
        with st.spinner('Searching for similar reviews...'):
            similar_reviews = find_similar_reviews(query)
            if similar_reviews:
                for idx, (review_id, score) in enumerate(similar_reviews, start=1):
                    st.write(f"{idx}. Review ID: {review_id} | Similarity: {score:.4f}")
            else:
                st.write("No similar reviews found.")
    else:
        st.write("Please enter a query to search for similar reviews.")
