# RL MCP Server Project

## Documentation

Current state of the program is as follows. It is a multi-LLM router that uses reinforcement learning (PPO) to 
decide which of three LLMs should answer each request. Everything is containerized so you can start/stop the
whole system with Docker Compose, observe it with Prometheus/Grafana, and (optionally) generate traffic with
Locust.

### Summary
It’s a learning router for three containerized LLMs that takes each incoming task, evaluates how well each 
model has been performing, and uses reinforcement learning logic to select the LLM most likely to give the 
best response. Over time, it keeps improving its choices based on experience—while you can watch the 
learning process unfold in Grafana.

### What Learning Means
- State: recent performance signals per model (mean, p95, queue proxy).

- Action: choose one of the 3 LLMs.

- Reward: currently negative latency (you can later add quality and cost terms).

- Update: PPO runs small online updates after gathering enough samples, so routing gradually shifts toward better models for the workload you’re sending.

### How a Request Flows
1. You (or Locust) send POST /infer to the RL Router.

2. The router constructs a quick “state” for each model: recent mean latency, p95 latency, and a small queue proxy (all from sliding windows).

3. PPO policy looks at those features and chooses an action = which model to call (A/B/C).

4. The router forwards the request to the chosen MCP adapter, which calls Ollama, which runs the actual LLM and returns text.

5. The router measures total latency and treats reward = –latency (faster is better).
It logs metrics and updates the PPO policy periodically, so it’s more likely to pick models that performed well under current conditions.

### Main Pieces
- RL Router (FastAPI + PPO): Receives user prompts, watches how fast each model responds, and learns which model to pick to minimize latency (and later, you can include quality/cost).

- Three LLM backends (MCP adapters): Lightweight HTTP services that expose /infer and call Ollama under the hood to run real models (e.g., Mistral, Llama 3, CodeLlama). Each adapter represents one “LLM choice”.

- Ollama: A local model server that actually runs the models. The adapters call Ollama’s API.

- Monitoring: Prometheus scrapes metrics; Grafana shows dashboards (throughput, p95 latency, routing distribution).

- (Optional) Locust: Sends a stream of prompts so the router has data to learn from.

### Why Adapters + Ollama
- The adapters give you a stable, uniform /infer API that your router already understands.

- Ollama makes running real models simple (CPU or GPU) without a heavy serving stack.
- Later, you can swap Ollama for a GPU-first server (e.g., vLLM) and keep the same router.

### Typical Lifecycle
1. Start the stack (Compose), pull models in Ollama (first time only).

2. Send prompts (manual or Locust) so PPO has data.

3. Watch dashboards to see routing and latency improve.

4. Stop the stack when done.

Note: unless you add a small checkpoint, PPO’s learned weights live in memory and reset when you docker compose down.

## Licence

The licence can be changed. By default this project has the [MIT Licence](./LICENCE).
<!-- You should update the year and name in the license file. -->
