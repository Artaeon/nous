# Nous

**Native Orchestration of Unified Streams**

> *"Nous (νοῦς) — the highest form of intellect; the faculty by which first principles are known."*
> — Aristotle, *Posterior Analytics*

A cognitive architecture that thinks, plans, reflects, and acts — running entirely on your machine. No cloud. No GPU. No API keys. Just bare metal and a local language model.

Nous is not a chatbot. It is a mind.

---

## Philosophy

Most AI tools are thin wrappers around API calls — prompt in, response out, nothing learned, nothing retained. They require cloud connectivity, GPU clusters, and monthly subscriptions to function.

Nous takes a different path. Inspired by cognitive science architectures (ACT-R, SOAR, Global Workspace Theory), it implements a **concurrent cognitive architecture** where independent mental processes — perception, reasoning, planning, execution, reflection, and learning — run as parallel streams, communicating through a shared blackboard.

The result: a system that doesn't just respond, but *thinks*. It decomposes goals, monitors its own reasoning, learns from experience, and adapts to the hardware it runs on.

Built in Go. Compiled to a single binary. Runs on anything with a CPU.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        BLACKBOARD                            │
│               (Shared Cognitive Workspace)                   │
│                                                              │
│  ┌───────────┐  ┌────────────┐  ┌─────────────────────┐     │
│  │  Percepts │  │  Working   │  │  Long-term Memory   │     │
│  │  (Input)  │  │  Memory    │  │  (mmap'd KV store)  │     │
│  └───────────┘  └────────────┘  └─────────────────────┘     │
│  ┌───────────┐  ┌────────────┐  ┌─────────────────────┐     │
│  │   Goals   │  │   Plans    │  │  Action History     │     │
│  │   Stack   │  │   Queue    │  │  (Episodic Memory)  │     │
│  └───────────┘  └────────────┘  └─────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
          ↕              ↕               ↕
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ Perceiver  │ │  Reasoner  │ │  Executor  │
   │  Stream    │ │  Stream    │ │  Stream    │
   └────────────┘ └────────────┘ └────────────┘
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ Reflector  │ │  Planner   │ │  Learner   │
   │  Stream    │ │  Stream    │ │  Stream    │
   └────────────┘ └────────────┘ └────────────┘
          ↕              ↕               ↕
   ┌──────────────────────────────────────────┐
   │            Ollama (Local LLM)            │
   │         qwen2.5 · phi3 · gemma2         │
   └──────────────────────────────────────────┘
```

### The Six Cognitive Streams

Each stream runs as a lightweight goroutine (~2KB), communicating through the blackboard via Go channels:

| Stream | Role | Cognitive Analogy |
|---|---|---|
| **Perceiver** | Parses input, extracts intent and entities | Sensory cortex |
| **Reasoner** | Chain-of-thought inference via local LLM | Prefrontal cortex |
| **Planner** | Decomposes goals into executable sub-tasks | Dorsolateral prefrontal |
| **Executor** | Runs tools and actions in the real world | Motor cortex |
| **Reflector** | Monitors reasoning quality, detects errors | Anterior cingulate |
| **Learner** | Extracts patterns, forms long-term memories | Hippocampus |

---

## Key Innovations

### Compressed Context Atoms

Instead of feeding the entire conversation history to the LLM on every turn (expensive on CPU), Nous compresses past context into dense **context atoms** — `{trigger → compressed_knowledge}` pairs. Before each inference call, only the most relevant atoms are retrieved and injected. This keeps the context window minimal, making CPU inference fast.

### Memory-Mapped Long-Term Knowledge

Long-term memory uses `mmap` — memory-mapped files managed by the OS kernel. No database, no serialization overhead, no startup cost. Cold data stays on disk until accessed. Hot data stays in RAM automatically. The system remembers everything across restarts with zero overhead.

### Adaptive Resource Budgeting

On startup, Nous detects available CPU cores, RAM, and thermal state. It dynamically adjusts:

- Fewer cores → fewer concurrent streams
- Less RAM → smaller context windows, lighter model selection
- Thermal throttling → reduced inference frequency

The system adapts to whatever hardware it runs on — from a Raspberry Pi to a workstation.

### Self-Reflective Reasoning

The Reflector stream continuously monitors the Reasoner's output for logical inconsistencies, hallucinations, and confidence drops. When it detects degraded reasoning, it triggers re-evaluation with a different decomposition strategy. The system doesn't just think — it thinks about its thinking.

---

## Recommended Models

Nous works with any Ollama-compatible model. Recommended for CPU-only inference:

| Model | Parameters | RAM Usage | Best For |
|---|---|---|---|
| `qwen2.5:1.5b` | 1.5B | ~1.5 GB | General reasoning, best quality/size ratio |
| `deepseek-r1:1.5b` | 1.5B | ~1.5 GB | Chain-of-thought, native reasoning traces |
| `phi3:mini` | 3.8B | ~3 GB | Complex tasks (requires 8GB+ RAM) |
| `qwen2.5:0.5b` | 0.5B | ~500 MB | Ultra-minimal, embedded systems |
| `smollm2:1.7b` | 1.7B | ~1.5 GB | Edge-optimized inference |

---

## Getting Started

### Prerequisites

- [Go 1.22+](https://go.dev/dl/)
- [Ollama](https://ollama.ai/) with a pulled model

### Install

```bash
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous
```

### Run

```bash
# Pull a model first
ollama pull qwen2.5:1.5b

# Start Nous
./nous
```

### Configuration

Nous auto-detects your hardware and selects optimal defaults. Override with flags:

```bash
./nous --model qwen2.5:1.5b --streams 4 --memory-path ~/.nous/memory
```

---

## Project Structure

```
nous/
├── cmd/nous/                  # Entry point
│   └── main.go
├── internal/
│   ├── blackboard/            # Shared cognitive workspace
│   ├── cognitive/             # The six cognitive streams
│   │   ├── perceiver.go
│   │   ├── reasoner.go
│   │   ├── planner.go
│   │   ├── executor.go
│   │   ├── reflector.go
│   │   └── learner.go
│   ├── memory/                # Working, long-term, episodic memory
│   ├── ollama/                # Ollama HTTP client (streaming)
│   ├── tools/                 # Extensible tool registry
│   └── compress/              # Context atom compression engine
├── go.mod
└── go.sum
```

---

## What Makes Nous Different

| | Cloud AI Tools | Nous |
|---|---|---|
| **Runtime** | Cloud API | Single local binary |
| **GPU** | Required | Not needed |
| **Architecture** | Prompt → Response | Concurrent cognitive streams |
| **Memory** | Stateless | Persistent, self-organizing |
| **Size** | N/A | ~10 MB binary |
| **Privacy** | Data sent to cloud | Fully local, air-gapped capable |
| **Adaptability** | Fixed | Adapts to hardware |
| **Self-awareness** | None | Reflective monitoring |

---

## Roadmap

- [x] Project scaffold and architecture design
- [ ] Ollama streaming client
- [ ] Blackboard shared state
- [ ] Perceiver + Reasoner (minimal thinking loop)
- [ ] Interactive CLI (REPL)
- [ ] Planner with hierarchical task decomposition
- [ ] Executor with tool system
- [ ] Reflector self-monitoring
- [ ] Learner pattern extraction
- [ ] Compressed context atoms
- [ ] Memory-mapped long-term storage
- [ ] Adaptive resource budgeting

---

## License

MIT

---

*Built by [Artaeon](mailto:raphael.lugmayr@stoicera.com)*
