import os
import json
import threading
import requests
import webview
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

class Api:
    def ask(self, question):
        context = get_context(question)
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": f"Answer questions about Ceaby using only this context:\n\n{context}. Do not mention that you're using a text for context. Do not take format directly from the given context. The language must be plain and concise. The format must seem natural, absolutely no asterisks in the text, but if you list, use bulletpoints. Do not go more than a paragraph or two. Avoid questions that are not about Ceaby. Ceaby goes by he and him."},
                {"role": "user", "content": question}
            ],
            "stream": False
        })
        return response.json()["message"]["content"]

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f0f1a;
    color: #fff;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  #header {
    padding: 18px 24px;
    background: #1a1a2e;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #2a2a3e;
  }

  #header h1 { font-size: 15px; font-weight: 600; letter-spacing: 0.3px; }

  #status {
    font-size: 12px;
    color: #4ade80;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  #status::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4ade80;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scroll-behavior: smooth;
  }

  #messages::-webkit-scrollbar { width: 4px; }
  #messages::-webkit-scrollbar-track { background: transparent; }
  #messages::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 4px; }

  .msg {
    display: flex;
    flex-direction: column;
    opacity: 0;
    transform: translateY(16px);
    animation: fadeUp 0.35s ease forwards;
  }

  @keyframes fadeUp {
    to { opacity: 1; transform: translateY(0); }
  }

  .msg.user { align-items: flex-end; }
  .msg.bot  { align-items: flex-start; }

  .bubble {
    max-width: 75%;
    padding: 11px 16px;
    border-radius: 18px;
    font-size: 14px;
    line-height: 1.6;
  }

  .msg.user .bubble {
    background: #1a6bb5;
    border-bottom-right-radius: 4px;
  }

  .msg.bot .bubble {
    background: #1e1e30;
    border-bottom-left-radius: 4px;
    color: #e0e0f0;
  }

  .label {
    font-size: 11px;
    color: #555570;
    margin-bottom: 4px;
    padding: 0 4px;
  }

  .typing span {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #555570;
    margin: 0 2px;
    animation: bounce 1.2s infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }

  #input-area {
    padding: 16px 20px;
    background: #1a1a2e;
    border-top: 1px solid #2a2a3e;
    display: flex;
    gap: 10px;
    align-items: center;
  }

  #input {
    flex: 1;
    background: #2b2d42;
    border: none;
    border-radius: 24px;
    padding: 12px 18px;
    font-size: 14px;
    color: #fff;
    outline: none;
    transition: background 0.2s;
  }

  #input:focus { background: #33354f; }
  #input::placeholder { color: #555570; }

  #send {
    background: #1a6bb5;
    border: none;
    border-radius: 50%;
    width: 42px; height: 42px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: transform 0.15s, background 0.15s;
    flex-shrink: 0;
  }

  #send:hover { background: #2280d6; transform: scale(1.05); }
  #send:active { transform: scale(0.95); }

  #send svg { width: 18px; height: 18px; fill: white; }
</style>
</head>
<body>

<div id="header">
  <h1>✦ Ask about Ceaby</h1>
  <div id="status">online</div>
</div>

<div id="messages"></div>

<div id="input-area">
  <input id="input" type="text" placeholder="Ask something about Ceaby..." />
  <button id="send">
    <svg viewBox="0 0 24 24"><path d="M2 21l21-9L2 3v7l15 2-15 2z"/></svg>
  </button>
</div>

<script>
  const messages = document.getElementById('messages');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');

  function addMessage(text, role) {
    const wrap = document.createElement('div');
    wrap.className = `msg ${role}`;

    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = role === 'user' ? 'You' : 'Ceaby';
    wrap.appendChild(label);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    wrap.appendChild(bubble);

    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
    return bubble;
  }

  function showTyping() {
    const wrap = document.createElement('div');
    wrap.className = 'msg bot';
    wrap.id = 'typing';

    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = 'Ceaby';
    wrap.appendChild(label);

    const bubble = document.createElement('div');
    bubble.className = 'bubble typing';
    bubble.innerHTML = '<span></span><span></span><span></span>';
    wrap.appendChild(bubble);

    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
  }

  function removeTyping() {
    const el = document.getElementById('typing');
    if (el) el.remove();
  }

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMessage(text, 'user');
    showTyping();

    const answer = await window.pywebview.api.ask(text);
    removeTyping();
    addMessage(answer, 'bot');
  }

  sendBtn.addEventListener('click', send);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

  window.addEventListener('pywebviewready', () => {
    addMessage("Hi! Ask me anything about Ceaby — projects, skills, experience.", 'bot');
  });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    api = Api()
    webview.create_window(
        "Ask about Ceaby",
        html=HTML,
        js_api=api,
        width=520,
        height=680,
        resizable=False
    )
    webview.start()

# "role": "system", "content": f"Answer questions about Ceaby using only this context:\n\n{context}. Do not mention that you're using a text for context. Do not take format directly from the given context. The language must be plain and concise. The format must seem natural, absolutely no asterisks in the text, but if you list, use bulletpoints. Do not go more than a paragraph or two."