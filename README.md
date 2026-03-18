<p align="center">
  <img src="assets/banner.svg" alt="Nous Banner" width="800">
</p>

<p align="center">
  <strong>Your Personal AI That Thinks, Learns, and Grows With You</strong>
</p>

<p align="center">
  <em>100% local. Zero cloud. Zero API keys. Fully private.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-0.9.0-blue?style=flat-square" alt="v0.9.0">
  <img src="https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white" alt="Go 1.22+">
  <img src="https://img.shields.io/badge/tools-45_built--in-orange?style=flat-square" alt="45 built-in tools">
  <img src="https://img.shields.io/badge/knowledge-669_chunks-blueviolet?style=flat-square" alt="669 knowledge chunks">
  <img src="https://img.shields.io/badge/virtual_context-66.9K_tokens-cyan?style=flat-square" alt="66.9K virtual tokens">
  <img src="https://img.shields.io/badge/binary-~14_MB-blue?style=flat-square" alt="~14 MB binary">
  <img src="https://img.shields.io/badge/deps-zero-brightgreen?style=flat-square" alt="Zero deps">
  <img src="https://img.shields.io/badge/cloud-not_required-green?style=flat-square" alt="No Cloud">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square" alt="MIT License">
</p>

---

> *"It is the active intellect that makes all things."*
> &mdash; Aristotle, *De Anima*, on nous (νοῦς)

---

## What Is Nous

Nous is an open-source **personal AI assistant** that runs **entirely on your local hardware** via [Ollama](https://ollama.ai). It's not a chatbot &mdash; it's a **cognitive system** that thinks, remembers, learns, and grows with you over time.

Built as a **concurrent cognitive architecture** with six independent processing streams, a deterministic NLU engine that handles most queries in under 1ms without any LLM call, 45 built-in tools, a knowledge engine with 669 encyclopedic chunks, and a virtual context system that makes a small model feel like it has 200K+ tokens of context.

**One ~14 MB Go binary. Zero dependencies. Zero cloud. Your data never leaves your machine.**

### What Nous Can Do

- **Understand you instantly** &mdash; deterministic NLU with 30+ intent categories routes queries in <1ms, most without any LLM call
- **Answer knowledge questions** &mdash; science, history, philosophy, technology, math, health, arts, and daily life topics from its 669-chunk knowledge base
- **Remember and grow** &mdash; learns your interests, preferences, and personal facts over time
- **Manage your day** &mdash; reminders, timers, tasks, calendar, routines, morning briefings, daily compass
- **Control your desktop** &mdash; volume, brightness, notifications, app launcher, screenshots
- **Search and research** &mdash; web search, URL fetching, RSS feeds, file exploration, semantic memory search
- **Work with files** &mdash; read, write, edit, grep, find, archive, disk usage &mdash; with undo support
- **Help with code** &mdash; run code (Python/Go/JS/Bash), 45 built-in tools, Go AST indexing
- **Translate and convert** &mdash; unit conversion, currency exchange, language translation, dictionary lookups
- **Network and system** &mdash; ping, DNS, port checks, process management, system info, hashing/encoding
- **Notes, todos, and email** &mdash; take notes, manage todo lists, check email
- **Generate QR codes** &mdash; create and read QR codes
- **Fine-tune itself** &mdash; learns from every interaction, can compile its experience into a custom model
- **Run as a server** &mdash; HTTP API + web UI for remote access
- **Deploy anywhere** &mdash; Docker, systemd, or one-line install

All running on CPU with a small local model. No API keys needed.

---

## Quick Start

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:1.5b
ollama pull nomic-embed-text    # for knowledge + episodic memory
ollama pull tinyllama            # optional: faster perception

# 2. Install Nous
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous

# 3. Load the knowledge base
./nous
/ingest knowledge/01_science.txt
/ingest knowledge/02_history.txt
/ingest knowledge/03_philosophy.txt
/ingest knowledge/04_technology.txt
/ingest knowledge/05_geography.txt
/ingest knowledge/06_mathematics.txt
/ingest knowledge/07_literature.txt
/ingest knowledge/08_health.txt
/ingest knowledge/09_arts_music.txt
/ingest knowledge/10_daily_life.txt
```

That's it. No `npm install`. No Python virtualenv. No API keys. No data leaves your machine.

---

## Architecture

```
                    ╔═══════════════════════════╗
                    ║   Virtual Context Engine   ║
                    ║   66.9K tokens across      ║
                    ║   knowledge + personal      ║
                    ╚═══════════╤═══════════════╝
                                │
              ╔═════════════════╧═══════════════════╗
              ║          Blackboard (shared state)    ║
              ╚══╤═══╤═══╤═══╤═══╤═══╤═════════════╝
                 │   │   │   │   │   │
           ┌─────┘   │   │   │   │   └─────┐
     ╔═════╧═════╗ ╔═╧═══╧═╗ ╔═╧═══╧═╗ ╔══╧════╗
     ║ Perceiver  ║ ║Reasoner║ ║Planner║ ║Learner║
     ║ (intent)   ║ ║(tools) ║ ║(steps)║ ║(grow) ║
     ╚═════╤═════╝ ╚═══╤═══╝ ╚═══╤═══╝ ╚══╤════╝
           │            │         │          │
     ╔═════╧═════╗ ╔═══╧═══╗ ╔══╧════╗ ╔══╧════════╗
     ║  Router    ║ ║Pipeline║ ║Execute║ ║  Neural    ║
     ║ (models)   ║ ║(fresh) ║ ║(run)  ║ ║  Cortex    ║
     ╚════════════╝ ╚═══════╝ ╚═══════╝ ╚═══════════╝
           │            │                      │
     ╔═════╧════════════╧══════════════════════╧═════╗
     ║              Memory System                      ║
     ║  Working · Long-term · Episodic · Project       ║
     ║  Knowledge Vec · Personal Growth                ║
     ╚════════════════════════════════════════════════╝
```

### Six Cognitive Streams

Each stream runs as an independent goroutine, communicating through the blackboard:

| Stream | Role | What It Does |
|--------|------|-------------|
| **Perceiver** | Sense | Extracts intent + entities from raw input |
| **Reasoner** | Think | Autonomous tool-calling agent (up to 8 steps) |
| **Planner** | Plan | Decomposes goals into step sequences |
| **Executor** | Act | Runs plan steps with tool calls |
| **Reflector** | Evaluate | Detects loops, hallucinations, quality issues |
| **Learner** | Grow | Extracts behavioral patterns, trains neural cortex |

### 4-Tier Query Classification

Not every question needs the full pipeline:

| Tier | Path | When | Latency |
|------|------|------|---------|
| **NLU Instant** | Deterministic NLU + tool dispatch | Weather, time, volume, translate, convert, timer, notes, todos, and 20+ more intents | **<1ms** (0 LLM calls) |
| **Fast** | Single LLM call | Greetings, thanks, yes/no | ~1-2s |
| **Medium** | LLM + knowledge context | Knowledge questions, explanations | ~3-5s |
| **Full** | Complete cognitive pipeline | Complex reasoning, multi-step tasks | ~5-15s |

---

## Key Innovations

### 1. Virtual Context Engine

**The innovation that makes a 4K-token model feel like 200K+.**

Instead of cramming everything into the context window, the Virtual Context Engine *weaves* the most relevant information from all sources for each query:

- **Knowledge Vec**: 669 encyclopedic chunks embedded with `nomic-embed-text`
- **Personal Growth**: Learned interests, preferences, and personal facts
- **Episodic Memory**: Every past interaction with semantic search
- **Working Memory**: Current session context with decay

Each source competes for context space. The weaver tracks what was useful and learns to allocate better over time.

```
  Total virtual context: 66.9K tokens
  Model context window:  4K tokens
  Budget per query:      1,500 tokens (most relevant slices)
```

### 2. Knowledge Vector Store

669 knowledge chunks across 10 domains, embedded and searchable in microseconds:

| Domain | Chunks | Topics |
|--------|--------|--------|
| Science | 83 | Physics, chemistry, biology, astronomy, earth science |
| History | 80 | Ancient through modern, key figures |
| Philosophy | 63 | Greek through 20th century, ethics |
| Technology | 70 | CS, networking, AI/ML, languages, hardware |
| Geography | 73 | Continents, countries, oceans, climate |
| Mathematics | 60 | Arithmetic through advanced, famous mathematicians |
| Literature | 60 | Classical through 20th century, literary concepts |
| Health | 60 | Human body, nutrition, conditions, wellness |
| Arts & Music | 60 | Visual arts, musicians, film, theater |
| Daily Life | 60 | Food, finance, productivity, communication, travel |

Add your own knowledge with `/ingest <file>`.

### 3. Personal Growth System

Nous learns about **you** over time:
- Tracks which topics you ask about most (frequency + recency weighted)
- Learns your communication style preferences
- Remembers personal facts you share
- Feeds personalized context into every response

### 4. Synthetic Neural Cortex

A pure Go feedforward neural network (64→32→N) that learns tool prediction:
- Trains on every successful interaction
- Predicts which tool to use before the LLM decides
- Xavier initialization, ReLU activation, softmax output, backpropagation
- Zero dependencies &mdash; implemented from scratch

### 5. Model Compiler

Compile Nous's learned experience into a custom Ollama model:
```
/compile    → Creates nous-v1, nous-v2, etc.
```
Embeds learned patterns, personality, and tool-calling behaviors into the model's system prompt via Ollama Modelfile.

### 6. Cognitive Pipeline (Fresh Context Per Step)

The #1 problem with small models: context fills up after 3-4 tool calls. Quality degrades catastrophically.

**Solution**: Each reasoning step gets a **fresh LLM conversation** with only the essentials. At step 8, context usage is ~15% instead of ~80%.

### 7. Deterministic NLU Engine

**Most queries never touch the LLM at all.**

A rule-based Natural Language Understanding engine with 30+ intent categories classifies user input in microseconds using pattern matching, word lists, and entity extraction. The NLU feeds an ActionRouter that dispatches directly to tools, returning results as DirectResponse without any LLM call.

```
"what's the weather?"        → NLU → weather tool → result    (0 LLM calls, <100ms)
"set a timer for 5 minutes"  → NLU → timer tool → started     (0 LLM calls, <1ms)
"translate hello to spanish" → NLU → translate tool → "hola"   (0 LLM calls, <500ms)
"convert 10 miles to km"     → NLU → convert tool → 16.09 km  (0 LLM calls, <1ms)
```

### 8. Cognitive Grounding (Anti-Hallucination)

Five layers preventing the model from making things up:

1. **Progressive Tool Disclosure** &mdash; 5-8 relevant tools per intent, not all 45
2. **Smart Truncation** &mdash; Tool-specific result shortening
3. **Result Validation** &mdash; Checks for empty reads, missing files, errors
4. **Context Budget** &mdash; Auto-compresses at 75%, forces answer at 85%
5. **Reflection Gate** &mdash; Detects loops, repetition, consecutive failures

### More Innovations

| Feature | What It Does |
|---------|-------------|
| **NLU Engine** | 30+ intent categories, deterministic routing, <1ms classification |
| **ActionRouter** | Direct tool dispatch from NLU, zero LLM calls for most queries |
| **Multi-Model Router** | Routes perception→tinyllama, reasoning→qwen2.5:1.5b |
| **Tool Choreography** | Records successful tool sequences as reusable recipes |
| **Predictive Cache** | Pre-computes likely follow-ups (read→test file, grep→read match) |
| **Filesystem Sentinel** | inotify-based real-time file watching |
| **Episodic Memory** | Remembers every interaction forever with semantic search |
| **Self-Improvement** | Collects training data, generates LoRA fine-tuning scripts |
| **Codebase Index** | Go AST parsing (~318 symbols), zero-cost structural context |
| **Prompt Distillation** | JIT-compiled prompts per query class (30-200 tokens vs 2000) |
| **Phantom Reasoning** | Pre-computes reasoning chains so LLM only writes conclusions |
| **Cognitive Firewall** | Validates outputs, blocks hallucinated tool calls |

---

## 45 Built-in Tools

| Category | Tools |
|----------|-------|
| **Explore** | `read`, `glob`, `grep`, `ls`, `tree` |
| **Modify** | `write`, `edit`, `patch`, `find_replace`, `replace_all`, `mkdir` |
| **System** | `shell`, `run`, `sysinfo`, `clipboard`, `fetch` |
| **Version Control** | `git`, `diff` |
| **Desktop** | `volume`, `brightness`, `notify`, `screenshot`, `app` |
| **Information** | `weather`, `dictionary`, `translate`, `websearch`, `rss`, `summarize` |
| **Productivity** | `notes`, `todos`, `calendar`, `email`, `timer`, `reminder` |
| **Convert & Compute** | `convert`, `currency`, `hash`, `qrcode`, `coderunner` |
| **Files & Storage** | `filefinder`, `archive`, `diskusage` |
| **Network & Processes** | `netcheck`, `process` |

---

## Memory System

| Layer | Scope | Persistence | Search |
|-------|-------|-------------|--------|
| **Working** | Current session | In-memory (decay) | Relevance scoring |
| **Long-term** | All sessions | JSON file | Keyword |
| **Project** | Per-project | `.nous/` directory | Keyword + exact |
| **Episodic** | Every interaction | `.nous/episodes.json` | **Semantic** (embeddings) |
| **Knowledge** | Encyclopedic | `.nous/knowledge.json` | **Vector** (cosine similarity) |
| **Personal** | User profile | `.nous/growth.json` | Topic + recency weighted |

---

## Slash Commands

### Daily Life
| Command | Description |
|---------|-------------|
| `/compass` | Triage panel: do now, focus, next anchor, risks |
| `/now` | One-line answer for the next best action |
| `/today` | Unread reminders and upcoming tasks |
| `/briefing` | Morning briefing &mdash; what matters today |
| `/checkin` | Quick pulse on your day |
| `/remind <when> <task>` | Create a persistent reminder |
| `/routine <daily\|weekdays> <HH:MM> <task>` | Recurring routine |

### Knowledge & Growth
| Command | Description |
|---------|-------------|
| `/ingest <file>` | Ingest a text file into the knowledge store |
| `/knowledge` | Knowledge store statistics |
| `/vctx` | Virtual context engine status |
| `/growth` | Personal growth profile |
| `/learn <fact>` | Teach Nous a personal fact |
| `/compile` | Compile experience into custom Ollama model |
| `/cortex` | Neural cortex statistics |

### Memory
| Command | Description |
|---------|-------------|
| `/memory` | Working memory contents |
| `/longterm` | Long-term memory entries |
| `/episodes` | Recent episodic memories |
| `/search <query>` | Semantic search through all memories |
| `/remember <k> <v>` | Store a project fact |
| `/recall <query>` | Search project memory |

### System
| Command | Description |
|---------|-------------|
| `/status` | Cognitive system status |
| `/tools` | List available tools |
| `/dashboard` | Full system overview |
| `/training` | Training data statistics |
| `/export <fmt>` | Export training data (jsonl/alpaca/chatml) |
| `/finetune` | Generate Modelfile + fine-tuning guide |
| `/undo` | Revert last file change |
| `/save [name]` | Save current session |
| `/sessions` | List saved sessions |
| `/help` | Full command reference |

---

## Usage

### Interactive REPL

```
$ ./nous

  ╭──────────────────────────────────────────────╮
  │ Nous v0.9.0                                  │
  │ model   qwen2.5:1.5b                         │
  │ host    http://localhost:11434                │
  │ tools   45 built-ins                          │
  │ memory  64 working slots                      │
  │ context 66.9K virtual tokens                  │
  ╰──────────────────────────────────────────────╯

  nous › what is the theory of relativity?

  The theory of relativity is a framework developed by Albert Einstein
  that explains the behavior of objects in motion and gravity. It
  consists of two main parts:

  1. Special Relativity (1905): introduced spacetime as curved and
     influenced by mass and energy.
  2. General Theory of Relativity (1915): massive objects cause
     space-time to curve, explaining gravity without Newtonian force.
  5.1s

  nous › good morning!

  Good morning to you too! How can I assist you today?
  3.0s

  nous › /compass
  ┌ Do Now ─────────────────────────┐
  │ Review your reminders           │
  └─────────────────────────────────┘
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Ollama model to use | `qwen2.5:1.5b` |
| `--host` | Ollama server address | `http://localhost:11434` |
| `--allow-shell` | Enable shell command execution | `false` |
| `--trust` | Skip confirmation prompts | `false` |
| `--serve` | Run as HTTP server | `false` |
| `--port` | HTTP server port | `3333` |
| `--public` | Bind server to `0.0.0.0` | `false` |
| `--resume` | Resume a previous session by ID | |
| `--memory` | Path for persistent memory | `~/.nous` |

### Server Mode

```bash
# HTTP API + Web UI
./nous --serve --port 3333 --allow-shell --trust

# API endpoints
curl -X POST http://localhost:3333/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "what is quantum mechanics?"}'
```

### Docker

```bash
docker compose up -d
# Open http://localhost:3333
```

---

## Fine-Tuning Your Own Model

Nous collects training data from every successful interaction:

```bash
# Inside Nous:
/training              # Check collected data
/export chatml         # Export as ChatML JSONL
/finetune              # Generate Modelfile + instructions
/compile               # Compile experience into Modelfile

# Then:
pip install unsloth transformers datasets peft trl
python .nous/finetune.py
ollama create nous-custom -f .nous/Modelfile
./nous --model nous-custom
```

This creates a model with Nous's personality, knowledge, and tool-calling patterns baked into the weights.

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
```

### Systemd (Production)

```bash
sudo make install
sudo systemctl enable --now nous
```

---

## Requirements

- **Go 1.22+** (build only)
- **Ollama** (runtime)
- **nomic-embed-text** (for knowledge + episodic memory)
- **Linux** (for inotify sentinel; rest works on macOS/Windows)
- **~2 GB RAM** (for qwen2.5:1.5b)
- GPU optional (CPU works fine)

---

## Project Structure

```
nous/
├── cmd/nous/
│   └── main.go                    # Entry point, REPL + server, 30+ slash commands
├── knowledge/                     # 10 encyclopedic knowledge files (669 chunks)
│   ├── 01_science.txt             # Physics, chemistry, biology, astronomy
│   ├── 02_history.txt             # Ancient through modern history
│   ├── 03_philosophy.txt          # Greek through 20th century philosophy
│   ├── 04_technology.txt          # CS, networking, AI/ML, hardware
│   ├── 05_geography.txt           # Continents, countries, climate
│   ├── 06_mathematics.txt         # Arithmetic through advanced math
│   ├── 07_literature.txt          # Classical through modern literature
│   ├── 08_health.txt              # Human body, nutrition, wellness
│   ├── 09_arts_music.txt          # Visual arts, music, film
│   └── 10_daily_life.txt          # Food, finance, productivity, travel
├── internal/
│   ├── cognitive/
│   │   ├── reasoner.go            # Core autonomous agent
│   │   ├── pipeline.go            # Fresh-context per step
│   │   ├── virtualctx.go          # Virtual Context Engine
│   │   ├── growth.go              # Personal Growth System
│   │   ├── knowledgevec.go        # Knowledge Vector Store
│   │   ├── neuralcortex.go        # Synthetic Neural Cortex
│   │   ├── modelcompiler.go       # Model Compiler (experience → weights)
│   │   ├── fastpath.go            # 3-tier query classification
│   │   ├── grounding.go           # Anti-hallucination system
│   │   ├── nlu.go                 # Deterministic NLU engine (30+ intents)
│   │   ├── action.go              # ActionRouter (tool dispatch, 0 LLM calls)
│   │   ├── router.go              # Multi-model routing
│   │   ├── recipes.go             # Tool choreography
│   │   ├── predictor.go           # Speculative pre-computation
│   │   ├── promptdistill.go       # JIT prompt compilation
│   │   ├── phantom.go             # Phantom reasoning chains
│   │   ├── persona.go             # Identity + system prompts
│   │   └── ...                    # 30+ cognitive modules
│   ├── memory/                    # 6-layer memory system
│   ├── sentinel/                  # inotify filesystem watcher
│   ├── server/                    # HTTP API + web UI
│   ├── training/                  # Self-improvement pipeline
│   ├── tools/                     # 45 built-in tools
│   └── ollama/                    # Ollama HTTP client
├── Dockerfile
├── docker-compose.yml
├── install.sh
└── go.mod                         # Zero external dependencies
```

---

## Documentation

| Document | What It Covers |
|----------|---------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Deep dive into cognitive architecture, key innovations, system diagrams |
| [TESTING.md](TESTING.md) | Test methodology, benchmarks, fuzz targets, coverage |
| [BENCHMARKS.md](BENCHMARKS.md) | Performance analysis: all operations profiled |
| [SECURITY.md](SECURITY.md) | Security model: defense layers, privacy guarantees, threat model |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We welcome contributions:

- New knowledge files (any domain!)
- Bug fixes and improvements
- New tools and capabilities
- Better prediction strategies
- Fine-tuning datasets and recipes

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ollama](https://ollama.ai) &mdash; local model inference
- [Qwen](https://github.com/QwenLM/Qwen2.5) &mdash; the remarkably capable 1.5B model
- [Nomic](https://nomic.ai) &mdash; embedding model for knowledge + memory
- Aristotle &mdash; for the name and the philosophy

---

<p align="center">
  <strong>Nous thinks, therefore Nous is &mdash; locally.</strong>
</p>
