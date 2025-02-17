# Airbnb Market Analysis Chatbot using Graph RAG & Generative AI

## Project Overview
This project involves creating a Generative AI-powered chatbot that answers questions about the Albany Airbnb market using Graph RAG architecture.
Data is stored and queried in a Neo4j graph database, with responses generated through text similarity embeddings. 
The system leverages a Streamlit interface for user interaction and utilizes pre-trained models for generating AI-driven responses.

## Features & Achievements
-  Graph RAG-based Chatbot: Uses Neo4j for efficient data retrieval and Generative AI for response generation.
-  Embedding Generation: Leverages Sentence Transformers to create vector embeddings from reviews.
-  Streamlit Interface: Allows users to query the chatbot for information on Albany Airbnb listings.

## Challenges Addressed
-  Efficient Data Retrieval: Implemented cosine similarity-based querying to fetch relevant data.
- Embedding Storage: Optimized embedding storage in Neo4j for fast retrieval during querying.

## Work in Progress
-  Model Optimization: Improving performance of the graph-based retrieval mechanism.
-  Scalability: Working on scaling the solution for larger datasets and real-time use cases.

## Next Steps
-  Extend to Edge Deployment: Explore deployment using NVIDIA/Qualcomm hardware accelerators like TensorRT and SNPE for improved inference performance.
-  Enhance Query Efficiency: Implement optimizations for faster retrieval in larger datasets.
-  Extend to Other Markets: Expand the system to support Airbnb data from different cities.

This project demonstrates the integration of Graph RAG and Generative AI for efficient question-answering systems, with potential applications in real-time data-driven chatbots for industries like hospitality and real estate.
