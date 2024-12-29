import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Define the find_similar_reviews function
def find_similar_reviews(query, top_n=5):
    # Encode the query into an embedding
    query_embedding = model.encode(query)

    # Query Neo4j for reviews with embeddings
    query_result = graph.run("MATCH (r:Review) WHERE r.embedding IS NOT NULL RETURN r.id, r.embedding LIMIT 1000")
    
    # Debugging: Print out the raw query results to see what is being returned
    results = list(query_result)
    st.write("Query Result:", results)

    similarities = []

    # Check if the results contain embeddings and process them
    for record in results:
        if 'r.embedding' in record:
            # Ensure the embedding is in the correct numeric format
            try:
                # If the embedding is a string representation of a list, convert it to a list
                review_embedding = np.array(eval(record['r.embedding']))  # Convert string to list if necessary
            except Exception as e:
                st.write(f"Error processing embedding for review {record['r.id']}: {e}")
                continue
            
            # Compute the cosine similarity between the query and the review embedding
            similarity = cosine_similarity([query_embedding], [review_embedding])
            similarities.append((record['r.id'], similarity[0][0]))

            # Debugging: Print the similarity score for each review
            st.write(f"Similarity with review {record['r.id']}: {similarity[0][0]}")

    # Debugging: Print the number of reviews with embeddings
    st.write(f"Retrieved {len(similarities)} reviews with embeddings.")

    # Sort the results by similarity score
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top N similar reviews
    return similarities[:top_n]
