import re
import os
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

model= SentenceTransformer("all-MiniLM-L6-v2")
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

def chunk_by_section(text):
    sections = re.split(r'\n(?=#{1,3} )', text)
    return [s.strip() for s in sections if s.strip()]

with open("about.md") as f:
    text = f.read()

chunks = chunk_by_section(text)

for chunk in chunks:
    embedding = model.encode(chunk).tolist()

    sb.table("documents").insert({
        "content": chunk,
        "embedding": embedding,
        "metadata": {"source": "about.md"}
    }).execute()