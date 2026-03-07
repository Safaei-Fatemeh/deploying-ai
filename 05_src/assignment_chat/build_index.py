# %%
import os
import pandas as pd
import chromadb
from openai import OpenAI

DATA_PATH = "data/notes.csv"
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "study_notes"

client = OpenAI(
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key="any value",
    default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")}
)

# %%
from dotenv import load_dotenv
load_dotenv("../../05_src/.secrets")

# %%
pd.read_csv(DATA_PATH).head()

# %%

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def main():
    df = pd.read_csv(DATA_PATH)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    for _, row in df.iterrows():
        doc_id = str(row["id"])
        title = str(row["title"])
        text = str(row["text"])
        source = str(row["source"])

        embedding = get_embedding(text)

        collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"title": title, "source": source}]
        )

    print(f"Indexed {len(df)} notes into ChromaDB at {CHROMA_PATH}")

# %%
main()

# %%



