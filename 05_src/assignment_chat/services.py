# %%
import os
import requests
import chromadb
from openai import OpenAI

CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "study_notes"

client = OpenAI(
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key="any value",
    default_headers={"x-api-key": str(os.getenv("API_GATEWAY_KEY"))}
)



# %%
from dotenv import load_dotenv
load_dotenv("../../05_src/.secrets")

# %%
def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_knowledge_base(query: str, n_results: int = 3):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    formatted_results = []
    for doc, meta in zip(documents, metadatas):
        formatted_results.append({
            "title": meta.get("title", "Untitled"),
            "text": doc,
            "source": meta.get("source", "unknown")
        })

    return formatted_results

def get_weather(city: str):
    # Step 1: geocode city name
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_response = requests.get(geo_url, params={"name": city, "count": 1}).json()

    if "results" not in geo_response or not geo_response["results"]:
        return f"Sorry, I could not find weather data for {city}."

    location = geo_response["results"][0]
    latitude = location["latitude"]
    longitude = location["longitude"]
    city_name = location["name"]
    country = location.get("country", "")

    # Step 2: get current weather
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_response = requests.get(
        weather_url,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": True
        }
    ).json()

    current = weather_response.get("current_weather", {})
    temperature = current.get("temperature")
    windspeed = current.get("windspeed")

    if temperature is None:
        return f"Sorry, I could not retrieve the weather for {city_name}."

    return f"The current weather in {city_name}, {country} is about {temperature}°C with wind speed around {windspeed} km/h."

def summarize_conversation(history):
    if not history:
        return "There is no conversation history yet."

    joined = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history])

    response = client.responses.create(
        model="gpt-4o",
        instructions="You are a helpful assistant. Summarize the conversation briefly and clearly.",
        input=joined
    )

    return response.output_text


