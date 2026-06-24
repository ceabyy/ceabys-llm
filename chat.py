import re
import os
import requests
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")
sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_ANON_KEY"))

def get_context(question):
    embedding = model.encode(question).tolist()
    result = sb.rpc("match_documents", {
        "query_embedding": embedding,
        "match_count": 5
    }).execute()
    return "\n\n".join([r["content"] for r in result.data])

def ask(question):
    context = get_context(question)
    response = requests.post("http://localhost:11434/api/chat", json={
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": f"Answer questions about Ceaby using only this context:\n\n{context} and try to make it as concise as possible, do not go beyond a paragraph in your response. Do not reference that you are basing your knowledge off of a text."},
            {"role": "user", "content": question}
        ],
        "stream": True
    }, stream=True)

    for line in response.iter_lines():
        if line:
            import json
            data = json.loads(line)
            chunk = data.get("message", {}).get("content", "")
            print(chunk, end="", flush=True)
    print()   

print("Hi! I'm an LLM that can answer questions about Ceaby. What would you like to know?")

while True:
    question = input("\nYou: ").strip()
    if not question:
        continue
    if question.lower() in ["exit", "quit"]:
        break
    print("\nAssistant: ", end="", flush=True)
    ask(question)
    print()