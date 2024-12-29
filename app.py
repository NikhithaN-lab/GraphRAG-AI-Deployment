import streamlit as st
from sentence_transformers import SentenceTransformer
from py2neo import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai  # For response generation (Fine-tuning)

# Neo4j connection setup
NEO4J_URI = "neo4j+s://32511ae0.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "HYKino3fm8r87dIde7v4FUZl0WPNHwCsXjzS6dlM4xI"
graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# SentenceTransformer model initialization
model = SentenceTransformer('all-MiniLM-L6-v2')

# OpenAI API setup (for LLM fine-tuning)
openai.api_key = 'your-openai-api-key'  # Replace with your OpenAI API key

# Streamlit UI setup
st.title('Airbnb Chatbot for Albany, NY')
st.write("Ask questions about Albany Airbnb listings and reviews:")

# User input
query = st.text_area("Enter your question:")

# Similarity Search Function
def find_similar_reviews(query, top_n=5):
    query_embedding = model.encode(query)
    
    # Query Neo4j for reviews with embeddings
    query_result = graph.run("MATCH (r:Review) WHERE r.embedding IS NOT NULL RETURN r.id, r.embedding LIMIT 1000")
    similarities = []
    
    for record in query_result:
        review_embedding = np.array(record['r.embedding'])
        similarity = cosine_similarity([query_embedding], [review_embedding])
        similarities.append((record['r.id'], similarity[0][0]))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Fine-tuning & Response Generation
def generate_response(query, similar_reviews):
    # Combine the similar reviews into a context for the LLM
    context = "\n".join([f"Review ID: {review_id}, Similarity: {similarity:.4f}" for review_id, similarity in similar_reviews])
    prompt = f"Given the following reviews and context, answer the user's question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # Call OpenAI GPT (or another LLM) for response generation using the new Chat API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can replace with another model like gpt-4 if available
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    
    return response['choices'][0]['message']['content'].strip()

# Process the user query
if st.button('Search Similar Reviews'):
    if query:
        # Retrieve similar reviews based on the query
        similar_reviews = find_similar_reviews(query)

        # Generate a response using the fine-tuned LLM based on similar reviews
        response = generate_response(query, similar_reviews)
        
        # Display the generated response
        st.write("### Answer:")
        st.write(response)

        # Display the top similar reviews
        st.write("### Top Similar Reviews:")
        for review_id, similarity in similar_reviews:
            st.write(f"Review ID: {review_id}, Similarity: {similarity:.4f}")
    else:
        st.write("Please enter a query to search.")
