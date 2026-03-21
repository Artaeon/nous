<p align="center">
  <img src="assets/banner.svg" alt="Nous Banner" width="100%">
</p>

<p align="center">
  <strong>Local-first cognitive infrastructure for reasoning, memory, and action without an LLM stack.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> |
  <a href="#why-nous-feels-different">Why Nous Feels Different</a> |
  <a href="ARCHITECTURE.md">Architecture</a> |
  <a href="BENCHMARKS.md">Benchmarks</a> |
  <a href="TESTING.md">Testing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-0f766e?style=flat-square" alt="v1.0.0">
  <img src="https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white" alt="Go 1.22+">
  <img src="https://img.shields.io/badge/runtime-one_static_binary-155e75?style=flat-square" alt="One static binary">
  <img src="https://img.shields.io/badge/latency-sub_15ms-0891b2?style=flat-square" alt="Sub 15ms pipeline">
  <img src="https://img.shields.io/badge/tools-51_local_tools-c2410c?style=flat-square" alt="51 local tools">
  <img src="https://img.shields.io/badge/memory-6_layers-0d9488?style=flat-square" alt="6-layer memory">
  <img src="https://img.shields.io/badge/LLM-none_required-16a34a?style=flat-square" alt="No LLM required">
  <img src="https://img.shields.io/badge/cloud-not_required-15803d?style=flat-square" alt="No cloud required">
  <img src="https://img.shields.io/badge/license-MIT-65a30d?style=flat-square" alt="MIT License">
</p>

---

> *"It is the active intellect that makes all things."*
> 
> Aristotle, *De Anima*, on nous

---

## What Nous Is

Nous is an open-source **cognitive system** that runs **entirely on your own machine**. It is not an interface bolted onto a remote model. It is a native stack for **understanding intent, planning discourse, reasoning over knowledge, using tools, and learning from use**.

The core idea is simple: if you want local AI to feel trustworthy, fast, and durable, you need more than offline inference. You need architecture. Nous builds that architecture directly in Go with a deterministic pipeline:

- **Deterministic NLU and routing** for fast tool dispatch and task classification
- **Thinking Engine** for frame selection and structured response generation
- **RST discourse planning** so answers are organized before text is written
- **Knowledge graph reasoning** for semantic lookup, causal links, and multi-hop chains
- **Compositional generation** using clause patterns, templates, Markov fluency, and embeddings
- **Persistent memory layers** that let the system accumulate useful context over time

The result is a system that writes, explains, compares, plans, searches, remembers, and operates your local environment from a single binary with **zero external dependencies, zero model downloads, and zero cloud requirement**.

## Why Nous Feels Different

Most "local AI" projects are still organized like hosted assistants: prompt in, model out, hope for the best. Nous is built around a different bet: **high-quality behavior can come from deliberate cognitive architecture, not just bigger inference.**

| Typical assistant stack | Nous |
|---|---|
| Remote or embedded model is the product | The cognitive system is the product |
| Prompt shaping is the control plane | Deterministic routing, frames, and planners are the control plane |
| Memory is usually bolt-on retrieval | Memory is a first-class subsystem with multiple layers |
| Tool use is an afterthought | Tool orchestration is part of the reasoning surface |
| Repeated tasks cost repeated inference | Crystals, recipes, and caches make repeated work faster |

That makes Nous useful in a different way: it behaves less like a black box and more like a compact, inspectable operating layer for personal intelligence.

## Innovation At A Glance

| System | What it contributes | Why it matters |
|---|---|---|
| **Thinking Engine** | Maps a query to one of 12 task types and a structural frame | Responses have intent-specific shape instead of generic prose |
| **Discourse Planner** | Uses Rhetorical Structure Theory to order sections and transitions | Prevents "fact salad" and improves coherence without an LLM |
| **Knowledge Graph** | 15 semantic relations, spreading activation, causal search | Enables reasoning chains instead of flat lookup |
| **Compositional Generator** | Clause patterns, realization strategies, Markov fluency, embeddings | Produces varied text from symbolic structure |
| **Response Crystals** | Stores successful outputs for semantic instant reuse | The system gets faster and more consistent with use |
| **Tool-Native Action Layer** | 51 built-in tools routed by deterministic NLU | Lets Nous act on the machine, not just talk about it |
| **Six-Layer Memory** | Working, long-term, episodic, project, knowledge, personal | Preserves context across sessions without external services |

## What Happens On Every Query

1. **Intent is classified** in under 1ms by the NLU engine.
2. **The router decides** whether the query is a direct tool action, cached response, or open-ended cognitive task.
3. **The Thinking Engine selects a frame** such as explanation, comparison, planning, or debate.
4. **The discourse planner organizes the answer** into rhetorical moves before text generation starts.
5. **Knowledge, memory, and tools are pulled in** as needed.
6. **The composer realizes the final answer** with deterministic generation strategies.

This is why Nous can answer a knowledge question, edit files, search memory, plan a project, or create a piece of writing without changing its architectural identity.

## What Nous Can Do

### Cognitive work

- Write emails, explanations, summaries, comparisons, plans, stories, and poetry
- Brainstorm options, debate tradeoffs, and give structured advice
- Answer questions from built-in knowledge and learned context

### Operate locally

- Route across 51 built-in tools for files, shell, desktop controls, system inspection, and information tasks
- Search the web, inspect URLs, read RSS, explore files, and index code
- Read, write, edit, patch, and undo file operations locally

### Remember and improve

- Track interests, preferences, project facts, and prior interactions
- Store episodic, project, and long-term memory in local files
- Reuse successful responses through crystals, recipes, and semantic caches

### Support daily life

- Manage reminders, routines, timers, notes, todos, journal entries, bookmarks, habits, and expenses
- Provide briefings, dashboards, and "what matters now" views
- Offer a local personal AI workflow without subscriptions or service lock-in

## System Snapshot

| Dimension | Details |
|---|---|
| **Binary** | One ~14 MB Go binary |
| **Dependencies** | Zero external runtime dependencies |
| **Latency** | Under 15ms total pipeline for cognitive generation paths |
| **Memory** | ~50 MB RAM, no model to load |
| **Interfaces** | REPL, HTTP server, slash commands, and channel integrations |
| **Privacy** | Your data stays on your machine unless you explicitly connect external services |

All local. All inspectable. All designed as a system, not a demo.

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
