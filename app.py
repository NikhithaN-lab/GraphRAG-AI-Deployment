import streamlit as st
from sentence_transformers import SentenceTransformer
from py2neo import Graph
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch  # For running the model on CPU/GPU

# Neo4j connection setup
NEO4J_URI = "neo4j+s://32511ae0.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "HYKino3fm8r87dIde7v4FUZl0WPNHwCsXjzS6dlM4xI"
graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# SentenceTransformer model initialization for embedding queries
model = SentenceTransformer('all-MiniLM-L6-v2')

# GPT-2 model initialization (via Hugging Face transformers)
gpt2_model_name = "gpt2"  # Example: GPT-2 model
tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = AutoModelForCausalLM.from_pretrained(gpt2_model_name)

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

# GPT-2 Model Response Generation
def generate_response_with_gpt2(query, similar_reviews):
    # Combine the similar reviews into a context for the GPT-2 model
    context = "\n".join([f"Review ID: {review_id}, Similarity: {similarity:.4f}" for review_id, similarity in similar_reviews])
    prompt = f"Given the following reviews and context, answer the user's question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # Tokenize the prompt with proper truncation and padding
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)

    # Generate the response using GPT-2 model
    with torch.no_grad():
        outputs = gpt2_model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

# Process the user query
if st.button('Search Similar Reviews'):
    if query:
        # Retrieve similar reviews based on the query
        similar_reviews = find_similar_reviews(query)

        # Generate a response using GPT-2 model based on similar reviews
        response = generate_response_with_gpt2(query, similar_reviews)
        
        # Display the generated response
        st.write("### Answer:")
        st.write(response)

        # Display the top similar reviews
        st.write("### Top Similar Reviews:")
        for review_id, similarity in similar_reviews:
            st.write(f"Review ID: {review_id}, Similarity: {similarity:.4f}")
    else:
        st.write("Please enter a query to search.")
