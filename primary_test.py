from openai import OpenAI
import json
import re
import time
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Tuple, Dict, Any
import traceback

ollama_session = requests.Session()
ollama_session.headers.update({"Content-Type": "application/json"})

SERVER_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3"


def ollama_call(payload):
    """Функция для вызова Ollama"""
    resp = ollama_session.post(SERVER_URL, json=payload, timeout=200)
    resp.raise_for_status()
    return resp.json()

mes = [{"role": "user", "content": "Привет"}]
payload = {
    "model": MODEL_NAME,
    "messages": mes,
    "stream": False,
    "options": {"temperature": 0, "seed": 12345, "think": False}
}

print(ollama_call(payload))