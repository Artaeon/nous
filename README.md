<h1 align="center">
  Nous
</h1>

<p align="center">
  <strong>Native Orchestration of Unified Streams</strong>
</p>

<p align="center">
  <em>An always-on local cognitive assistant with persistent memory, background reminders, and zero cloud dependencies.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.6.0-blue?style=flat-square" alt="v0.6.0">
  <img src="https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white" alt="Go 1.22+">
  <img src="https://img.shields.io/badge/tests-316_passing-brightgreen?style=flat-square" alt="316 tests">
  <img src="https://img.shields.io/badge/binary-~10_MB-blue?style=flat-square" alt="~10 MB binary">
  <img src="https://img.shields.io/badge/deps-zero-brightgreen?style=flat-square" alt="Zero deps">
  <img src="https://img.shields.io/badge/cloud-not_required-green?style=flat-square" alt="No Cloud">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square" alt="MIT License">
</p>

---

> *"It is the active intellect that makes all things."*
> &mdash; Aristotle, *De Anima*, on nous

---

## What Is Nous

Nous is an open-source **personal AI assistant** that runs **entirely on your local hardware** via [Ollama](https://ollama.ai). No cloud. No API keys. No data leaves your machine.

It's built as a **concurrent cognitive architecture**: six independent processing streams (perceive, reason, plan, execute, reflect, learn) communicate through a shared blackboard, producing intelligent behavior from the interplay of simple, well-defined modules.

The result is a **single ~10 MB Go binary** with zero external dependencies that can:

- Remember personal preferences and tasks across restarts
- Run background reminders and recurring daily check-ins
- Generate recurring routines such as morning reviews and weekday inbox sweeps
- Act as a local assistant for notes, files, shell commands, and web lookups
- Still help with code, files, and terminal workflows when you need it
- Chain up to 8 tool calls autonomously per turn
- Remember every interaction forever (episodic memory with semantic search)
- Learn successful tool sequences and replay them (tool choreography)
- Watch your filesystem in real-time (inotify sentinel)
- Speculatively pre-compute likely follow-up results (predictive cache)
- Fine-tune itself from its own experience (LoRA training pipeline)
- Run as an HTTP server with a web UI
- Deploy anywhere with Docker, systemd, or a one-line install

All running on CPU with a 1.5B parameter model.

---

## Quick Start

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:1.5b
ollama pull tinyllama   # optional: faster perception

# 2. Install Nous
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous

# 3. Run
./nous
```

That's it. No `npm install`. No Python virtualenv. No API keys.

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous
./nous --version
```

### One-Line Install (Linux)

```bash
curl -sSL https://raw.githubusercontent.com/artaeon/nous/main/install.sh | bash
```

### Docker

```bash
docker compose up -d
# Open http://localhost:3333
```

### Server Mode

```bash
# Run as HTTP API with web UI
./nous --serve --port 3333

# Expose on your LAN only when you mean to
./nous --serve --public --port 3333
```

---

## Usage

### Interactive REPL

```
$ ./nous --allow-shell

                +===================================+
                |             N O U S               |
                |   Native Orchestration of         |
                |       Unified Streams             |
                +===================================+

  version 0.6.0 | amd64 | 16 cores | 32 GB RAM

  connecting to ollama... OK (qwen2.5:1.5b)
  routing: perception->tinyllama, reasoning->qwen2.5:1.5b
  scanning project... nous (Go, 56 files)
  codebase index: 350 symbols
  sentinel: watching 12 dirs
  6 cognitive streams active
  18 tools: read, write, edit, grep, glob, ls, ...

  I am Nous. I think, therefore I am - locally.

  nous> /today
  nous> /remind tomorrow 09:00 call dentist
  nous> summarize what matters today
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Ollama model to use | `qwen2.5:1.5b` |
| `--host` | Ollama server address | `http://localhost:11434` |
| `--allow-shell` | Enable shell command execution | `false` |
| `--trust` | Skip confirmation prompts | `false` |
| `--serve` | Run as HTTP server | `false` |
| `--listen` | HTTP listen host (with `--serve`) | `127.0.0.1` |
| `--public` | Bind server to `0.0.0.0` | `false` |
| `--port` | HTTP server port | `3333` |
| `--resume` | Resume a previous session by ID | |
| `--memory` | Path for persistent memory | `~/.nous` |

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/compass` | Compact triage panel for next action, focus, and risks |
| `/now` | Deterministic answer for what to do next |
| `/focus` | Personalized focus guidance |
| `/checkin` | Quick pulse on your day |
| `/prep` | Prepare for the next meeting or time-boxed task |
| `/today` | Show unread reminders and upcoming assistant tasks |
| `/tasks` | List pending assistant tasks |
| `/routines` | List recurring assistant routines |
| `/remind <when> <task>` | Create a persistent reminder |
| `/routine <daily\|weekdays> <HH:MM> <task>` | Create a recurring assistant routine |
| `/done <task-id>` | Mark a task as completed |
| `/pref <k> <v>` | Store a personal preference |
| `/capture <note>` | Save a personal note for future assistant context |
| `/prefs` | List saved preferences |
| `/status` | Cognitive system status |
| `/memory` | Working memory contents |
| `/longterm` | Long-term memory entries |
| `/episodes` | Recent episodic memories |
| `/search <query>` | Semantic search through all memories |
| `/remember <k> <v>` | Store a project fact |
| `/recall <query>` | Search project memory |
| `/forget <key>` | Remove a project fact |
| `/training` | Training data statistics |
| `/export <fmt>` | Export training data (jsonl/alpaca/chatml) |
| `/finetune` | Generate Modelfile + fine-tuning guide |
| `/undo` | Revert last file change |
| `/history` | Show undo stack |
| `/tools` | List available tools |
| `/sessions` | List saved sessions |
| `/save [name]` | Save current session |
| `/clear` | Clear conversation context |
| `/quit` | Exit (auto-saves) |

### Assistant HTTP API

When running with `--serve`, Nous now exposes assistant-first endpoints in addition to `/api/chat` and `/api/jobs`:

| Endpoint | Description |
|----------|-------------|
| `GET /api/assistant/today` | Return unread reminders, today items, and upcoming tasks |
| `GET/POST /api/assistant/tasks` | List or create persistent tasks |
| `POST /api/assistant/tasks/{id}/done` | Mark a task as complete |
| `GET/POST /api/assistant/preferences` | List or update saved preferences |
| `GET/POST /api/assistant/routines` | List or create recurring routines |
| `POST /api/assistant/notifications/read` | Mark unread reminders as read |

---

## Architecture

```
                         +-------------------+
                         |    Blackboard     |
                         |  (shared state)   |
                         +--------+----------+
                                  |
          +-----------+-----------+-----------+-----------+
          |           |           |           |           |
    +-----------+ +----------+ +---------+ +----------+ +----------+
    | Perceiver | | Reasoner | | Planner | | Executor | | Reflector|
    | (intent)  | | (tools)  | | (steps) | | (run)    | | (eval)   |
    +-----------+ +----------+ +---------+ +----------+ +----------+
          |           |                                       |
    +-----------+ +----------+                          +----------+
    |  Router   | | Pipeline |                          | Learner  |
    | (models)  | | (fresh)  |                          | (pattern)|
    +-----------+ +----------+                          +----------+
```

### Six Cognitive Streams

Each stream runs as an independent goroutine, communicating through the blackboard:

| Stream | Role | Event |
|--------|------|-------|
| **Perceiver** | Extracts intent + entities from raw input | `percept` |
| **Reasoner** | Autonomous tool-calling agent (up to 8 steps) | `percept` &rarr; `last_answer` |
| **Planner** | Decomposes goals into step sequences | `goal_pushed` |
| **Executor** | Runs plan steps with tool calls | `plan_set` |
| **Reflector** | Evaluates reasoning quality | `action_recorded` |
| **Learner** | Extracts behavioral patterns | `goal_updated` |

### 18 Built-in Tools

| Category | Tools |
|----------|-------|
| **Explore** | `read`, `glob`, `grep`, `ls`, `tree` |
| **Modify** | `write`, `edit`, `patch`, `find_replace`, `replace_all`, `mkdir` |
| **System** | `shell`, `run`, `sysinfo`, `clipboard`, `fetch` |
| **Version Control** | `git`, `diff` |

### Four Memory Layers

| Layer | Scope | Persistence | Search |
|-------|-------|-------------|--------|
| **Working** | Current session | In-memory (decay) | Relevance scoring |
| **Long-term** | All sessions | JSON file | Keyword |
| **Project** | Per-project | `.nous/` directory | Keyword + exact |
| **Episodic** | Every interaction ever | `.nous/episodes.json` | **Semantic** (embeddings) |

---

## Key Innovations

### Cognitive Pipeline (Fresh Context Per Step)

The #1 problem with small models: context window fills up after 3-4 tool calls, and quality degrades catastrophically.

**Solution**: Each reasoning step gets a **fresh LLM conversation** with only:
1. Compact system prompt
2. Original user question
3. One-line compressed summaries of previous steps
4. Latest tool result

At step 8, context usage is **~15%** instead of ~80% with message accumulation.

### Multi-Model Router

Auto-discovers all locally available Ollama models and routes by cognitive task:

| Task | Model | Latency |
|------|-------|---------|
| Perception | tinyllama | ~200ms |
| Reasoning | qwen2.5:1.5b | ~2s |
| Compression | tinyllama | ~200ms |
| Reflection | tinyllama | ~200ms |

Falls back to single model if only one is available.

### Cognitive Grounding (Anti-Hallucination)

Five layers preventing the model from making things up:

1. **Progressive Tool Disclosure** &mdash; Show 5-8 relevant tools per intent (not all 18)
2. **Smart Truncation** &mdash; Tool-specific result shortening
3. **Result Validation** &mdash; Checks for empty reads, missing files, permission errors
4. **Context Budget** &mdash; Tracks token usage, auto-compresses at 75%, forces answer at 85%
5. **Reflection Gate** &mdash; Detects loops, repetition, consecutive failures; forces convergence

### Filesystem Sentinel

Linux inotify-based ambient file watching with zero polling:
- Watches all project directories recursively
- Debounced batched notifications (500ms window)
- Auto-updates codebase index when `.go` files change
- Ignores `.git`, `vendor`, swap files, etc.

### Tool Choreography (Learned Recipes)

Records successful multi-step tool sequences and replays them:
- "grep for function → read the file" becomes a reusable recipe
- Keyword-based matching finds relevant recipes for new queries
- Confidence scoring based on success rate
- Parameterized replay with `$FILE`/`$DIR` placeholders

### Predictive Pre-computation

After each tool call, speculatively pre-executes likely follow-ups:
- `read X.go` &rarr; pre-cache `X_test.go`
- `grep` &rarr; pre-read first matched file
- `ls` &rarr; pre-read README/main entry point
- Only pre-computes read-only tools (never writes)

### Episodic Memory with Semantic Search

Every interaction is stored forever with embedding vectors:
- Cosine similarity search over Ollama's embedding space
- Hybrid search: semantic first, keyword fallback
- Success rate tracking, tool usage statistics
- Auto-extracted topic tags

### Self-Improvement Pipeline

Nous can fine-tune its own base model:

```
Interactions → Quality Filter → Export (ChatML/Alpaca/JSONL)
                                       ↓
                               LoRA Fine-tuning (unsloth)
                                       ↓
                               Custom Ollama Model
                                       ↓
                            nous --model nous-custom
```

The `/finetune` command generates everything needed: Modelfile, training data export, and a complete Python fine-tuning script.

### Codebase Index

Go AST parsing extracts every function, struct, method, interface, constant, and variable with full signatures. Gives the model structural context in ~50 tokens instead of reading files.

---

## Deployment

### Local (Development)

```bash
./nous                          # Interactive REPL
./nous --allow-shell            # Enable shell commands
./nous --trust --allow-shell    # No confirmation prompts
```

### Server (HTTP API)

```bash
./nous --serve --port 3333 --allow-shell --trust
# Web UI at http://localhost:3333
# API at http://localhost:3333/api/chat
```

### Docker

```bash
docker compose up -d
# Includes Ollama with GPU passthrough
```

### Systemd (Production)

```bash
sudo make install
sudo systemctl enable --now nous
# Running at http://localhost:3333
```

### API Example

```bash
# Chat
curl -X POST http://localhost:3333/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "show me the main function"}'

# Status
curl http://localhost:3333/api/status

# Health check
curl http://localhost:3333/api/health
```

---

## Fine-Tuning Your Own Model

Nous collects training data from every successful interaction. When you have enough data, you can fine-tune a custom model:

```bash
# Inside Nous:
/training              # Check how many pairs collected
/export chatml         # Export as ChatML JSONL
/finetune              # Generate Modelfile + instructions

# Then:
pip install unsloth transformers datasets peft trl
python .nous/finetune.py
ollama create nous-custom -f .nous/Modelfile
./nous --model nous-custom
```

This creates a model with Nous's personality and tool-calling patterns baked into the weights &mdash; not just prompts.

---

## Project Structure

```
nous/
├── cmd/nous/
│   └── main.go                 # Entry point, REPL + server mode
├── internal/
│   ├── blackboard/             # Shared cognitive workspace (pub/sub)
│   ├── cognitive/
│   │   ├── reasoner.go         # Core autonomous agent
│   │   ├── pipeline.go         # Fresh-context per step
│   │   ├── grounding.go        # Anti-hallucination system
│   │   ├── router.go           # Multi-model routing
│   │   ├── recipes.go          # Tool choreography
│   │   ├── predictor.go        # Speculative pre-computation
│   │   ├── perceiver.go        # Intent extraction
│   │   ├── planner.go          # Goal decomposition
│   │   ├── executor.go         # Plan execution
│   │   ├── reflector.go        # Quality evaluation
│   │   ├── learner.go          # Pattern extraction
│   │   ├── tool_selector.go    # Progressive tool disclosure
│   │   ├── persona.go          # Identity + system prompts
│   │   ├── conversation.go     # Message management
│   │   ├── session.go          # Session persistence
│   │   └── scanner.go          # Project auto-detection
│   ├── index/
│   │   └── codebase.go         # Go AST structural index
│   ├── memory/
│   │   ├── working.go          # Decay-based working memory
│   │   ├── longterm.go         # Persistent KV store
│   │   ├── project.go          # Per-project fact store
│   │   ├── episodic.go         # Interaction replay + embeddings
│   │   └── undo.go             # File modification undo stack
│   ├── sentinel/
│   │   └── watcher.go          # inotify filesystem watcher
│   ├── server/
│   │   └── server.go           # HTTP API + web UI
│   ├── training/
│   │   ├── collector.go        # Training data pipeline
│   │   └── modelfile.go        # Ollama Modelfile generator
│   ├── tools/
│   │   ├── builtin.go          # 18 built-in tools
│   │   └── registry.go         # Tool registry
│   ├── compress/
│   │   └── atoms.go            # Context compression
│   └── ollama/
│       └── client.go           # Ollama HTTP client
├── Dockerfile
├── docker-compose.yml
├── nous.service                # systemd unit
├── install.sh                  # One-line installer
├── Makefile
├── LICENSE
├── CONTRIBUTING.md
└── go.mod                      # Zero external dependencies
```

---

## Comparison: Nous vs Claude Code

| | Claude Code | Nous |
|---|---|---|
| **Model** | Claude Opus (200B+) | qwen2.5:1.5b (1.5B) |
| **Privacy** | Cloud API | **100% local** |
| **Cost** | $15-75/month | **Free forever** |
| **Binary** | ~50 MB + Node.js | **~10 MB, zero deps** |
| **Startup** | 2-5 seconds | **<1 second** |
| **Memory** | Per-project CLAUDE.md | **4-layer memory + semantic search** |
| **Self-improvement** | None | **LoRA fine-tuning pipeline** |
| **File watching** | None | **inotify sentinel** |
| **Predictive cache** | None | **Speculative pre-computation** |
| **Tool learning** | None | **Recipe choreography** |
| **Server mode** | No | **HTTP API + web UI** |
| **Docker** | No | **docker-compose with Ollama** |
| **Reasoning depth** | Exceptional | Good (8 steps, pipeline-optimized) |
| **Code generation** | Best-in-class | Focused edits (1.5B model) |

**Nous wins on**: privacy, cost, architecture, memory, self-improvement, deployment flexibility.

**Claude Code wins on**: raw model intelligence (200B vs 1.5B).

As local models improve, Nous's architecture amplifies those gains. A 7B model in this architecture is genuinely competitive.

---

## Requirements

- **Go 1.22+** (build only)
- **Ollama** (runtime)
- **Linux** (for inotify sentinel; rest works on macOS/Windows)
- **~2 GB RAM** (for qwen2.5:1.5b)
- GPU optional (CPU works fine)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

We welcome contributions! Whether it's:
- Bug fixes and improvements
- New tools
- Support for more languages in the codebase index
- Better prediction strategies
- Fine-tuning datasets and recipes
- Documentation and examples

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ollama](https://ollama.ai) &mdash; local model inference
- [Qwen](https://github.com/QwenLM/Qwen2.5) &mdash; the remarkably capable 1.5B model
- Aristotle &mdash; for the name and the philosophy

---

<p align="center">
  <strong>Nous thinks, therefore Nous is.</strong>
</p>
