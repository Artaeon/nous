<p align="center">
  <img src="assets/banner.svg" alt="Nous Banner" width="800">
</p>

<p align="center">
  <strong>The AI That Thinks Without an LLM</strong>
</p>

<p align="center">
  <em>Pure cognitive engine. Zero LLM. Zero cloud. Zero API keys. Fully local.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue?style=flat-square" alt="v1.0.0">
  <img src="https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white" alt="Go 1.22+">
  <img src="https://img.shields.io/badge/LLM-none_required-brightgreen?style=flat-square" alt="No LLM">
  <img src="https://img.shields.io/badge/tools-51_built--in-orange?style=flat-square" alt="51 built-in tools">
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

Nous is an open-source **personal AI** that runs **entirely on your local hardware** with **zero LLM dependency**. It's not a chatbot wrapper &mdash; it's a **pure cognitive engine** that thinks, reasons, composes, remembers, and grows with you over time.

Built from the ground up with a **Thinking Engine** (intent classification → frame selection → section generation), **Rhetorical Structure Theory discourse planning**, **knowledge graph reasoning**, **compositional text generation**, **Markov chains**, **word embeddings**, and a **conversational learning engine** &mdash; all in pure Go. No neural network inference. No external model. No API calls.

**One ~14 MB Go binary. Zero dependencies. Zero cloud. Your data never leaves your machine.**

### How Is This Possible?

Traditional AI assistants are just LLM wrappers &mdash; they send your text to a neural network and hope for the best. Nous takes a fundamentally different approach:

1. **Thinking Engine** &mdash; Classifies your intent (12 task types, 35+ signal patterns), selects a structural frame, generates each section with purpose
2. **Discourse Planner** &mdash; Plans rhetorical structure before generating text (RST-based: hook → define → elaborate → conclude), producing coherent prose instead of fact salad
3. **Knowledge Graph** &mdash; 15 semantic relations, spreading activation search, multi-hop reasoning chains, causal inference
4. **Compositional Generation** &mdash; 10 clause patterns, 6 realization strategies, tone-aware phrase pools, Markov chains for fluency
5. **Word Embeddings** &mdash; 50-dimensional vectors for semantic word selection, taxonomy-seeded + co-occurrence trained

The result: Nous writes emails, brainstorms ideas, explains concepts, compares options, gives advice, tells stories, writes poetry, plans projects, and debates topics &mdash; all without a single LLM call.

### What Nous Can Do

- **Think and compose** &mdash; write emails, brainstorm, explain concepts, compare options, plan projects, give advice, debate, create stories and poetry
- **Understand you instantly** &mdash; deterministic NLU with 30+ intent categories routes queries in <1ms
- **Answer knowledge questions** &mdash; science, history, philosophy, technology, math, health, arts, and daily life from its knowledge graph
- **Remember and grow** &mdash; learns your interests, preferences, and personal facts over time
- **Manage your day** &mdash; reminders, timers, tasks, calendar, routines, morning briefings, daily compass
- **Control your desktop** &mdash; volume, brightness, notifications, app launcher, screenshots
- **Search and research** &mdash; web search, URL fetching, RSS feeds, file exploration, semantic memory search
- **Work with files** &mdash; read, write, edit, grep, find, archive, disk usage &mdash; with undo support
- **Help with code** &mdash; run code (Python/Go/JS/Bash), Go AST indexing, 51 built-in tools
- **Translate and convert** &mdash; unit conversion, currency exchange, language translation, dictionary lookups
- **Network and system** &mdash; ping, DNS, port checks, process management, system info, hashing/encoding
- **Life management** &mdash; journal, habits, expenses, bookmarks, notes, todos, passwords
- **Fine-tune its knowledge** &mdash; learns from every conversation, knowledge packages for instant domain expertise

All instant. All local. All in one binary.

---

## Quick Start

```bash
# 1. Install Nous
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous

# 2. Run
./nous
```

That's it. No `npm install`. No Python virtualenv. No API keys. No model downloads. No GPU required.

---

## Architecture

```
                    ╔═══════════════════════════════╗
                    ║     Thinking Engine            ║
                    ║   intent → frame → generate    ║
                    ╚═══════════╤═══════════════════╝
                                │
           ╔════════════════════╧════════════════════╗
           ║        Discourse Planner (RST)           ║
           ║   schema → sections → transitions        ║
           ╚════╤══════════╤════════════╤════════════╝
                │          │            │
     ╔══════════╧══╗ ╔════╧═════╗ ╔═══╧═════════╗
     ║  Knowledge   ║ ║ Composer  ║ ║  Generative  ║
     ║  Graph       ║ ║ (tone,    ║ ║  Engine      ║
     ║  (15 rels)   ║ ║  style)   ║ ║  (Markov,    ║
     ║              ║ ║           ║ ║   templates)  ║
     ╚══════╤══════╝ ╚═════╤════╝ ╚══════╤═══════╝
            │              │              │
     ╔══════╧══════════════╧══════════════╧══════════╗
     ║              NLU → ActionRouter                ║
     ║   30+ intents · 51 tools · <1ms routing        ║
     ╚══════╤══════════════════════════════╤══════════╝
            │                              │
     ╔══════╧══════════════════════════════╧══════════╗
     ║              Memory System                      ║
     ║  Working · Long-term · Episodic · Project       ║
     ║  Knowledge Graph · Personal Growth · Embeddings ║
     ╚════════════════════════════════════════════════╝
```

### Cognitive Pipeline

Every query flows through a deterministic pipeline:

| Stage | Component | What It Does | Latency |
|-------|-----------|-------------|---------|
| 1 | **NLU Engine** | Intent classification + entity extraction | <1ms |
| 2 | **ActionRouter** | Dispatches to 51 tools or Thinking Engine | <1ms |
| 3 | **Thinking Engine** | Task classification → frame selection | <1ms |
| 4 | **Discourse Planner** | RST schema → section planning | <1ms |
| 5 | **Composer** | Knowledge graph + Markov + templates → text | <10ms |

**Total: under 15ms for any query. Zero external calls.**

### The Thinking Engine

12 task types, each with specialized generation:

| Task | Triggered By | Output |
|------|-------------|--------|
| **Compose** | "write an email to..." | Full email with greeting, body, closing, signoff |
| **Brainstorm** | "brainstorm ideas for..." | Categorized ideas with synthesis |
| **Analyze** | "explain...", "what is..." | Hook → definition → mechanism → significance |
| **Teach** | "teach me about..." | Goal → prerequisites → steps → tips |
| **Advise** | "should I...", "help me decide..." | Empathy → analysis → suggestions → encouragement |
| **Compare** | "compare X vs Y" | Intro → item descriptions → differences → verdict |
| **Summarize** | "summarize...", "TL;DR" | Overview → key points → conclusion |
| **Create** | "write a story/poem about..." | Narrative arc or verse with imagery |
| **Plan** | "plan a...", "how should I approach..." | Objective → phases → considerations → timeline |
| **Debate** | "argue for/against..." | Thesis → evidence → counterpoints → conclusion |
| **Reflect** | "what do you think about..." | Thoughtful analysis with multiple perspectives |
| **Converse** | General chat | Natural conversational response |

### Frame System

12 structural templates ensure coherent output:

```
Email Frame:      greeting → opening → body → closing → signoff
Brainstorm Frame: context → ideas (categorized) → synthesis
Explanation Frame: hook → definition → mechanism → example → significance
Comparison Frame:  intro → item_a → item_b → differences → verdict
Tutorial Frame:    goal → prerequisites → steps → tips → next_steps
```

Each frame section is filled by the appropriate generator with knowledge graph facts, tone-aware phrases, and compositional text.

---

## Key Innovations

### 1. Discourse Planning (RST)

**The single biggest quality improvement for text generation without an LLM.**

Based on Rhetorical Structure Theory (Mann & Thompson, 1988), the discourse planner:
1. Analyzes available facts to determine rhetorical moves
2. Selects a discourse schema (explanatory, narrative, comparative, brief, feature-focus, origin-focus)
3. Plans sections with communicative goals and transition phrases
4. The generative engine fills each section in order

This is the difference between "fact salad" and "coherent prose."

### 2. Knowledge Graph Reasoning

15 semantic relations enable multi-hop reasoning:
- `is_a`, `described_as`, `has`, `part_of`, `used_for`
- `created_by`, `founded_by`, `founded_in`, `located_in`
- `related_to`, `similar_to`, `causes`, `follows`, `offers`

Spreading activation search finds relevant facts across the graph. The causal engine traces cause-effect chains. The reasoning engine builds multi-step inference chains.

### 3. Compositional Text Generation

6 realization strategies produce varied, natural text:
- **Subject-verb-object** with modifiers and prepositional phrases
- **Topic-comment** structures for explanatory text
- **Existence/classification** patterns for definitions
- **Cause-effect** patterns for reasoning
- **Temporal/sequential** patterns for narratives

Combined with trigram Markov chains for fluency and 50-dim word embeddings for semantic word selection.

### 4. Conversational Learning

Nous learns from every interaction:
- Extracts facts, entities, and relationships from conversations
- Grows its knowledge graph organically
- Tracks your interests and communication preferences
- Knowledge packages provide instant domain expertise

### 5. 51 Built-in Tools

Deterministic NLU routes to tools in <1ms:

| Category | Tools |
|----------|-------|
| **Explore** | `read`, `glob`, `grep`, `ls`, `tree` |
| **Modify** | `write`, `edit`, `patch`, `find_replace`, `replace_all`, `mkdir` |
| **System** | `shell`, `run`, `sysinfo`, `clipboard`, `fetch` |
| **Version Control** | `git`, `diff` |
| **Desktop** | `volume`, `brightness`, `notify`, `screenshot`, `app` |
| **Information** | `weather`, `dictionary`, `translate`, `websearch`, `rss`, `summarize` |
| **Productivity** | `notes`, `todos`, `calendar`, `email`, `timer`, `reminder` |
| **Life Management** | `journal`, `habits`, `expenses`, `bookmarks`, `passwords` |
| **Convert & Compute** | `convert`, `currency`, `hash`, `qrcode`, `coderunner`, `calculator` |
| **Files & Storage** | `filefinder`, `archive`, `diskusage` |
| **Network & Processes** | `netcheck`, `process` |

### 6. Personal Growth System

Nous learns about **you** over time:
- Tracks which topics you ask about most (frequency + recency weighted)
- Learns your communication style preferences
- Remembers personal facts you share
- Feeds personalized context into every response

---

## Memory System

| Layer | Scope | Persistence | Search |
|-------|-------|-------------|--------|
| **Working** | Current session | In-memory (decay) | Relevance scoring |
| **Long-term** | All sessions | JSON file | Keyword |
| **Project** | Per-project | `.nous/` directory | Keyword + exact |
| **Episodic** | Every interaction | `.nous/episodes.json` | Keyword + semantic |
| **Knowledge Graph** | Structured knowledge | `.nous/cognitive_graph.json` | Spreading activation |
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
| `/packages` | List loaded knowledge packages |
| `/knowledge` | Knowledge store statistics |
| `/growth` | Personal growth profile |
| `/learn <fact>` | Teach Nous a personal fact |
| `/cortex` | Neural cortex statistics |

### Memory
| Command | Description |
|---------|-------------|
| `/memory` | Working memory contents |
| `/longterm` | Long-term memory entries |
| `/episodes` | Recent episodic memories |
| `/search <query>` | Search through all memories |
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
  │ Nous v1.0.0                                  │
  │ engine  cognitive                             │
  │ tools   51 built-ins                          │
  │ memory  64 working slots                      │
  ╰──────────────────────────────────────────────╯

  nous › write an email to my boss about taking Friday off

  Dear Boss,

  I hope this message finds you well. I am writing regarding
  taking Friday off. When it comes to this matter, there are
  several aspects worth considering.

  Thank you for your time and consideration.
  I look forward to hearing from you.

  Best regards,
  0.008s

  nous › brainstorm ideas for a birthday party

  Let's explore ideas for a birthday party.

  Approaches:
  • Consider the venue and theme options
  • Plan activities and entertainment

  Creative angles:
  • Personalized touches that reflect the guest of honor
  • Interactive elements for guest engagement

  The most promising direction combines personalization
  with interactive experiences.
  0.003s
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
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

## Knowledge Packages

Extend Nous's knowledge instantly with JSON knowledge packages:

```bash
# Place packages in ./packages/
ls packages/
  general_knowledge.json
  programming.json
  cooking.json

# Nous loads them automatically on startup
./nous
  loaded 3 knowledge packages (2400 facts, 800 vocab)
```

Create your own packages with facts, vocabulary, and semantic relations.

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
- **Linux** (for inotify sentinel; rest works on macOS/Windows)
- **~50 MB RAM** (no model to load)
- No GPU required
- No external services required

---

## Project Structure

```
nous/
├── cmd/nous/
│   └── main.go                    # Entry point, REPL + server, 30+ slash commands
├── packages/                      # Knowledge packages (JSON)
├── internal/
│   ├── cognitive/
│   │   ├── thinking.go            # Thinking Engine (12 task types)
│   │   ├── discourse.go           # RST Discourse Planner
│   │   ├── frames.go              # Frame System (12 structural templates)
│   │   ├── composer.go            # Compositional text generation
│   │   ├── generative.go          # Markov + template + clause generation
│   │   ├── embeddings.go          # 50-dim word embeddings
│   │   ├── cognitive_graph.go     # Knowledge graph (15 relations)
│   │   ├── reasoning_chain.go     # Multi-hop reasoning
│   │   ├── nlu.go                 # Deterministic NLU (30+ intents)
│   │   ├── action.go              # ActionRouter (51 tools, 0 LLM calls)
│   │   ├── semantic.go            # Semantic analysis engine
│   │   ├── learning_engine.go     # Conversational learning
│   │   ├── personal_response.go   # Personalized responses
│   │   ├── factstore.go           # Extractive fact store
│   │   ├── tracker.go             # Conversation tracking
│   │   └── ...                    # 40+ cognitive modules
│   ├── memory/                    # 6-layer memory system
│   ├── sentinel/                  # inotify filesystem watcher
│   ├── server/                    # HTTP API + web UI
│   ├── training/                  # Training data collection
│   ├── tools/                     # 51 built-in tools
│   └── simd/                      # SIMD-optimized vector operations
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

- New knowledge packages (any domain!)
- Bug fixes and improvements
- New tools and capabilities
- Better generation strategies
- New discourse schemas and frames

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Aristotle &mdash; for the name and the philosophy of active intellect
- Mann & Thompson &mdash; for Rhetorical Structure Theory
- The Go team &mdash; for a language that makes this possible in one binary

---

<p align="center">
  <strong>Nous thinks, therefore Nous is &mdash; no LLM required.</strong>
</p>
