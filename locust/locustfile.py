# =============================
# File: locust/locustfile.py
# =============================
from locust import HttpUser, task, between
import random

PROMPTS = [
    "Explain the significance of the Doppler effect.",
    "Write a Python function to reverse a linked list.",
    "Summarize the causes of the French Revolution in 3 bullets.",
    "Generate unit tests for the following function: def add(a,b): return a+b",
]

class MCPClient(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(5)
    def infer(self):
        p = random.choice(PROMPTS)
        self.client.post("/infer", json={"prompt": p, "max_new_tokens": 64})