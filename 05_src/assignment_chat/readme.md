# Assignment 2 - StudyBuddy Chat

## Overview
This project implements a simple conversational AI system called **StudyBuddy**.  
It uses a Gradio chat interface and provides three services:

1. **Weather Service (API Call)**  
   The user can ask for weather information by typing a message such as `weather Toronto`.  
   The app uses the Open-Meteo API and rewrites the result in natural language.

2. **Semantic Search Service**  
   The user can search a small local knowledge base by typing a message such as `search embeddings`.  
   The app uses embeddings and a persistent ChromaDB collection to retrieve the most relevant notes.

3. **Conversation Summary Service**  
   The user can type `summarize chat` to get a short summary of the conversation so far.

## Personality
The chat assistant is called **StudyBuddy**.  
It is designed to be friendly, helpful, and slightly academic.

## Memory
The system keeps recent conversation history and includes it in normal chat responses.

## Guardrails
The system refuses:
- requests to reveal or modify the system prompt
- questions about cats or dogs
- questions about horoscopes or zodiac signs
- questions about Taylor Swift

## Files
- `app.py`: main Gradio chat app
- `services.py`: backend services
- `build_index.py`: script used to create the ChromaDB index
- `data/notes.csv`: small dataset used for semantic search
- `chroma_store/`: persistent ChromaDB storage

## Notes
The embeddings were generated using the course API gateway and stored in a persistent ChromaDB collection.
