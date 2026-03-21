<p align="center">
  <img src="assets/banner.svg" alt="Nous Banner" width="100%">
</p>

<p align="center">
  <strong>A fully generative local AI built from scratch in pure Go. No LLM. No cloud. No dependencies.</strong>
</p>

<p align="center">
  <a href="#what-makes-nous-different">What Makes Nous Different</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#live-demo">Live Demo</a> |
  <a href="ARCHITECTURE.md">Architecture</a> |
  <a href="BENCHMARKS.md">Benchmarks</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-0f766e?style=flat-square" alt="v1.0.0">
  <img src="https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white" alt="Go 1.22+">
  <img src="https://img.shields.io/badge/runtime-one_static_binary-155e75?style=flat-square" alt="One static binary">
  <img src="https://img.shields.io/badge/intent_classification-38%C2%B5s-0891b2?style=flat-square" alt="38µs intent classification">
  <img src="https://img.shields.io/badge/tools-51_built--in-c2410c?style=flat-square" alt="51 built-in tools">
  <img src="https://img.shields.io/badge/neural_model-537_KB-7c3aed?style=flat-square" alt="537 KB neural model">
  <img src="https://img.shields.io/badge/LLM-none_required-16a34a?style=flat-square" alt="No LLM required">
  <img src="https://img.shields.io/badge/cloud-not_required-15803d?style=flat-square" alt="No cloud required">
  <img src="https://img.shields.io/badge/license-MIT-65a30d?style=flat-square" alt="MIT License">
</p>

---

> *"It is the active intellect that makes all things."*
>
> Aristotle, *De Anima*, on nous (νοῦς)

---

## What Nous Is

Nous is a **fully generative AI** that runs entirely on your machine. It understands what you say, reasons about it, and generates natural language responses &mdash; all without calling an LLM, downloading a model, or touching the cloud.

It does this through a cognitive architecture built from first principles in Go:

- A **neural intent classifier** trained on ~3,600 examples classifies your intent in 38 microseconds
- A **knowledge graph** with 15 semantic relations enables multi-hop reasoning
- A **compositional text generator** produces varied natural language from symbolic structures
- A **6-layer memory system** remembers you across sessions
- **51 built-in tools** let it act on your machine, not just talk about it

The result: a personal AI that writes, explains, remembers, reasons, and operates your local environment from a single binary. Every response is generated at runtime. Nothing is canned.

## What Makes Nous Different

Nous is not a wrapper around someone else's model. It is a complete cognitive system.

|  | ChatGPT / Claude | Llama / Ollama | Nous |
|---|---|---|---|
| **Where it runs** | Cloud API | Local (4-16 GB RAM, GPU recommended) | **Local (50 MB RAM, any CPU)** |
| **Model size** | Hundreds of GB | 4-70 GB | **537 KB** |
| **Intent classification** | 500ms-3s | 100ms+ | **38 microseconds** |
| **Privacy** | Data leaves your machine | Local but heavy | **100% local, zero network** |
| **Dependencies** | API key, internet | Python, CUDA, model download | **Zero. One Go binary.** |
| **Deterministic** | No (temperature, sampling) | No | **Yes** |
| **Cost** | $20/month+ | Free but resource-heavy | **Free and lightweight** |
| **Learns from you** | Per-session only | No | **Yes, persistently** |

### The Innovation

Most "local AI" means running a smaller version of a cloud model on your hardware. Nous takes a fundamentally different approach: **deliberate cognitive architecture instead of brute-force inference**.

Instead of billions of parameters doing everything, Nous has specialized systems that each do one thing well:

```
  Input: "what is quantum physics?"

  1. Neural classifier  →  intent: explain (0.99 confidence, 38µs)
  2. Knowledge graph     →  finds: physics → natural science, has quantum mechanics,
                             general relativity, thermodynamics, used for understanding
                             forces and energy
  3. Discourse planner   →  selects explanation schema, plans sections
  4. Composer            →  generates coherent prose from structured facts

  Output: Natural language explanation with hook, definition, key properties,
          and significance. Generated in <15ms. Zero API calls.
```

This is why Nous can run on a Raspberry Pi and still give you thoughtful, structured answers.

---

## Live Demo

```
$ ./nous

  ╭──────────────────────────────────────────────╮
  │ Nous v1.0.0                                  │
  │ model   cognitive                             │
  │ tools   51 built-ins                          │
  │ memory  64 working slots                      │
  ╰──────────────────────────────────────────────╯

  nous › hello
  Hey there Raphael.

  nous › define serendipity
  serendipity (noun): A combination of events which have come together
  by chance to make a surprisingly good or wonderful outcome.

  nous › translate hello to french
  hello → Bonjour (French)

  nous › who made you?
  I was created by Artaeon — built from scratch in pure Go as a fully
  local AI that runs entirely on your machine.

  nous › good morning
  Good morning! It's Saturday, March 21.

  Weather: clear sky, 13.7°C, humidity 73%, wind 6.3 km/h

  Habits:
  [x] meditation (daily)
  [ ] exercise (daily)

  Tasks:
  [ ] #1 review pull request
  [ ] #2 buy milk

  nous › remind me to call mom tomorrow
  Reminder set: "call mom tomorrow" in 18h (fires at 09:00)

  nous › remember my favorite color is blue
  Got it — I'll remember that your favorite color is blue.

  nous › what is my favorite color?
  blue

  nous › what is quantum physics?
  Physics is one of those things that rewards a closer examination.
  Put simply, physics is a natural science. Several things define it:
  quantum mechanics, general relativity, and thermodynamics.
  Physics powers our understanding of forces and energy.

  nous › write me a poem about the ocean
  in the ocean we find the love of all things
  in the grove of the ocean, something stirs

  nous › set a timer for 5 minutes
  Timer started: 5m0s (fires at 15:24)

  nous › generate a password
  Password: S{zMC5mj4M0mM^N_

  nous › take a screenshot
  Screenshot saved: /tmp/nous-screenshot.png

  nous › bye
  Take care, Raphael.
```

Every response above was generated in real time. No canned responses. No API calls. No internet needed.

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous

# Run
./nous
```

That's it. No `npm install`. No Python virtualenv. No API keys. No model downloads. No GPU.

On first launch, Nous trains its neural classifier (~70 seconds). Subsequent launches load the saved model instantly.

---

## Architecture

```
                         ╔══════════════════════════════════╗
                         ║   Neural Intent Classifier        ║
                         ║   2-layer MLP · 49 intents        ║
                         ║   38µs · 99.7% accuracy           ║
                         ╚═══════════════╤══════════════════╝
                                         │
              ╔══════════════════════════╧═══════════════════════╗
              ║                  Action Router                    ║
              ║          51 tools · intent → action dispatch      ║
              ╚══════╤══════════╤════════════╤═══════════╤══════╝
                     │          │            │           │
          ╔══════════╧══╗ ╔════╧═════╗ ╔═══╧══════╗ ╔═╧══════════╗
          ║  Thinking    ║ ║ Knowledge ║ ║ Creative  ║ ║  Tools      ║
          ║  Engine      ║ ║ Graph     ║ ║ Engine    ║ ║  (51)       ║
          ║  12 frames   ║ ║ 15 rels   ║ ║ poems,    ║ ║  weather,   ║
          ║  brainstorm, ║ ║ spreading ║ ║ stories,  ║ ║  files,     ║
          ║  explain,    ║ ║ activation║ ║ jokes,    ║ ║  translate, ║
          ║  compare,    ║ ║ causal    ║ ║ reflec-   ║ ║  reminder,  ║
          ║  plan ...    ║ ║ chains    ║ ║ tions     ║ ║  timer ...  ║
          ╚══════╤══════╝ ╚═════╤════╝ ╚═══╤══════╝ ╚════╤═══════╝
                 │              │           │              │
          ╔══════╧══════════════╧═══════════╧══════════════╧══════╗
          ║              Discourse Planner (RST)                    ║
          ║    schema → sections → transitions → coherent prose     ║
          ╚══════════════════════════╤══════════════════════════════╝
                                    │
          ╔═════════════════════════╧═══════════════════════════════╗
          ║              Compositional Generator                      ║
          ║   Markov chains · clause patterns · embeddings · tone    ║
          ╚══════════════════════════╤═══════════════════════════════╝
                                    │
          ╔═════════════════════════╧═══════════════════════════════╗
          ║              Memory System                                ║
          ║  Working · Long-term · Episodic · Project · Knowledge     ║
          ║  Personal growth · Response crystals · Learned patterns   ║
          ╚═══════════════════════════════════════════════════════════╝
```

### Neural Intent Classifier

The brain of the routing system. A 2-layer MLP (multilayer perceptron) that classifies user intent across 49 categories in 38 microseconds.

| Property | Value |
|---|---|
| Architecture | 2048 input → 64 hidden (ReLU) → 49 output (softmax) |
| Feature extraction | Character n-grams + word unigrams via FNV-32 hashing trick |
| Training | SGD with momentum (0.9), cosine annealing LR, 80 epochs |
| Training accuracy | 99.7% on ~3,600 augmented examples |
| Novel input accuracy | 81% on completely unseen phrasings |
| Inference speed | 38µs average (10,000x faster than an API call) |
| Model size | 537 KB binary format |
| vs pattern matching | 5x faster, higher accuracy, better generalization |

The classifier trains from scratch on first launch using auto-generated data from the NLU's word lists. The trained model is saved to disk and loaded instantly on subsequent runs.

### Cognitive Pipeline

Every query flows through a deterministic pipeline:

| Stage | Component | What It Does | Latency |
|---|---|---|---|
| 1 | **Neural Classifier** | Intent classification + confidence scoring | 38µs |
| 2 | **Entity Extractor** | Intent-specific argument parsing | <1ms |
| 3 | **Action Router** | Dispatches to 51 tools or cognitive engines | <1ms |
| 4 | **Thinking Engine** | Frame selection for open-ended queries | <1ms |
| 5 | **Discourse Planner** | RST schema for coherent text organization | <1ms |
| 6 | **Composer** | Knowledge + Markov + templates → natural language | <10ms |

**Total: under 15ms for any query. Zero external calls.**

### Thinking Engine

12 task types, each with specialized generation:

| Task | Example Trigger | Output |
|---|---|---|
| **Compose** | "write an email about..." | Greeting → body → closing → signoff |
| **Brainstorm** | "brainstorm ideas for..." | Categorized ideas with synthesis |
| **Analyze** | "explain...", "what is..." | Hook → definition → mechanism → significance |
| **Teach** | "teach me about..." | Goal → prerequisites → steps → tips |
| **Advise** | "should I...", "help me..." | Empathy → analysis → suggestions |
| **Compare** | "compare X vs Y" | Descriptions → differences → verdict |
| **Summarize** | "summarize...", "TL;DR" | Overview → key points → conclusion |
| **Create** | "write a poem about..." | Verse with imagery and rhythm |
| **Plan** | "plan a...", "how to..." | Phases → milestones → timeline |
| **Debate** | "argue for/against..." | Thesis → evidence → counterpoints |
| **Reflect** | "what is the meaning of..." | Multiple perspectives, thoughtful analysis |
| **Converse** | General chat | Natural, contextual response |

### Knowledge Graph

15 semantic relations enable multi-hop reasoning:

```
                    ┌─── is_a ──── natural science
                    │
  quantum physics ──┼─── has ───── quantum mechanics
                    │              general relativity
                    │              thermodynamics
                    │
                    ├─── used_for ── understanding forces and energy
                    │
                    └─── created_by ── Isaac Newton (classical)
                                       Albert Einstein (relativistic)
```

Relations: `is_a`, `described_as`, `has`, `part_of`, `used_for`, `created_by`, `founded_by`, `founded_in`, `located_in`, `related_to`, `similar_to`, `causes`, `follows`, `offers`, `instance_of`

The graph supports spreading activation search, causal chain inference, and Wikipedia batch imports for expanding domain knowledge.

### Creative Engine

Pure-Go creative text generation:

- **Poems**: free verse, haiku, quatrain, limerick, sonnet
- **Stories**: narrative arc with setting, conflict, resolution
- **Jokes**: setup-punchline with topic integration
- **Reflections**: philosophical musings from knowledge graph
- **Fun facts**: random interesting facts from the knowledge base

### Memory System

| Layer | What It Stores | How It Persists |
|---|---|---|
| **Working** | Current context, recent topics | In-memory with decay |
| **Long-term** | Personal facts, preferences | JSON file |
| **Episodic** | Every interaction with timing | JSON file |
| **Project** | Per-directory project context | `.nous/` directory |
| **Knowledge** | Structured facts and relations | Cognitive graph |
| **Personal** | Interests, growth, style | Growth profile |
| **Crystals** | Successful response patterns | Semantic cache |

---

## 51 Built-in Tools

Deterministic NLU routes to the right tool in microseconds:

| Category | Tools |
|---|---|
| **Explore** | `read`, `glob`, `grep`, `ls`, `tree` |
| **Modify** | `write`, `edit`, `patch`, `find_replace`, `replace_all`, `mkdir` |
| **System** | `shell`, `run`, `sysinfo`, `clipboard`, `fetch` |
| **Desktop** | `volume`, `brightness`, `notify`, `screenshot`, `app` |
| **Information** | `weather`, `dictionary`, `translate`, `websearch`, `rss`, `summarize` |
| **Productivity** | `notes`, `todos`, `calendar`, `email`, `timer`, `reminder` |
| **Life** | `journal`, `habits`, `expenses`, `bookmarks`, `passwords` |
| **Compute** | `convert`, `currency`, `hash`, `qrcode`, `coderunner`, `calculator` |
| **Files** | `filefinder`, `archive`, `diskusage` |
| **Network** | `netcheck`, `process`, `git`, `diff` |

---

## Benchmarks

| Metric | Nous | ChatGPT | Llama 3 (8B) |
|---|---|---|---|
| Intent classification | **38µs** | ~800ms | ~200ms |
| Full response generation | **<15ms** | 1-5s | 0.5-2s |
| Memory footprint | **~50 MB** | Cloud | 4-16 GB |
| Model file size | **537 KB** | Cloud | 4-8 GB |
| Privacy | **100% local** | Cloud | Local |
| Deterministic | **Yes** | No | No |
| Dependencies | **Zero** | Internet + API key | Python + CUDA |
| Learns from you | **Yes** | No | No |
| Cost | **Free** | $20/mo+ | Free (hardware cost) |

### Neural Classifier Benchmarks

```
Training:      3,600 examples, 49 intents, 99.7% accuracy
Inference:     38µs average (10,000 iterations)
Model size:    537 KB
Novel inputs:  81% accuracy on 121 completely unseen phrasings
vs patterns:   100% accuracy vs 97% (5x faster)
```

---

## Slash Commands

### Daily Life
| Command | Description |
|---|---|
| `/compass` | Triage panel: do now, focus, next anchor, risks |
| `/now` | One-line answer for the next best action |
| `/today` | Unread reminders and upcoming tasks |
| `/briefing` | Morning briefing &mdash; weather, tasks, habits, schedule |
| `/checkin` | Quick pulse on your day |
| `/remind <when> <task>` | Create a persistent reminder |

### Knowledge & Growth
| Command | Description |
|---|---|
| `/packages` | List loaded knowledge packages |
| `/knowledge` | Knowledge store statistics |
| `/growth` | Personal growth profile |
| `/learn <fact>` | Teach Nous a personal fact |
| `/cortex` | Neural cortex statistics |

### Memory
| Command | Description |
|---|---|
| `/memory` | Working memory contents |
| `/longterm` | Long-term memory entries |
| `/search <query>` | Search through all memories |
| `/remember <k> <v>` | Store a personal fact |
| `/recall <query>` | Search memory |

### System
| Command | Description |
|---|---|
| `/status` | Cognitive system status |
| `/tools` | List available tools |
| `/dashboard` | Full system overview |
| `/training` | Training data statistics |
| `/export <fmt>` | Export training data (jsonl/alpaca/chatml) |
| `/undo` | Revert last file change |
| `/help` | Full command reference |

---

## Installation

### From Source (Recommended)

```bash
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous
./nous
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

### CLI Flags

| Flag | Description | Default |
|---|---|---|
| `--allow-shell` | Enable shell command execution | `false` |
| `--trust` | Skip confirmation prompts | `false` |
| `--serve` | Run as HTTP server | `false` |
| `--port` | HTTP server port | `3333` |
| `--public` | Bind server to `0.0.0.0` | `false` |
| `--memory` | Path for persistent memory | `~/.nous` |

### Server Mode

```bash
./nous --serve --port 3333 --allow-shell

# API
curl -X POST http://localhost:3333/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "what is quantum mechanics?"}'
```

---

## Knowledge Packages

Extend Nous's knowledge with JSON knowledge packages:

```bash
# Place packages in ./packages/
ls packages/
  general_knowledge.json
  programming.json
  cooking.json

# Nous loads them on startup
./nous
  loaded 3 knowledge packages (2400 facts, 800 vocab)
```

Import from Wikipedia:

```
nous › /import-wikidata philosophy 500
  Importing philosophy from Wikidata (limit 500)...
  Imported 487 facts into philosophy
  Knowledge graph: 12,400 nodes, 31,200 edges
```

---

## Requirements

- **Go 1.22+** (build only)
- **~50 MB RAM** (no model to load)
- No GPU required
- No external services required
- Works on Linux, macOS, and Windows

---

## Project Structure

```
nous/
├── cmd/nous/
│   └── main.go                      # Entry point, REPL, server, 30+ slash commands
├── packages/                        # Knowledge packages (JSON)
├── internal/
│   ├── cognitive/
│   │   ├── neural_classifier.go     # 2-layer MLP intent classifier (38µs)
│   │   ├── neural_features.go       # Feature extraction (hashing trick)
│   │   ├── neural_nlu.go            # Neural NLU integration + training data
│   │   ├── nlu.go                   # NLU engine (49 intents, entity extraction)
│   │   ├── action.go                # ActionRouter (51 tools, 0 LLM calls)
│   │   ├── thinking.go              # Thinking Engine (12 task types)
│   │   ├── discourse.go             # RST Discourse Planner
│   │   ├── frames.go                # Frame System (12 structural templates)
│   │   ├── creative.go              # Creative engine (poems, stories, jokes)
│   │   ├── composer.go              # Compositional text generation
│   │   ├── generative.go            # Markov + template generation
│   │   ├── embeddings.go            # 50-dim word embeddings
│   │   ├── cognitive_graph.go       # Knowledge graph (15 relations)
│   │   ├── reasoning_chain.go       # Multi-hop reasoning
│   │   ├── causal.go                # Causal inference engine
│   │   ├── learning_engine.go       # Conversational learning
│   │   ├── personal_response.go     # Personalized responses
│   │   └── ...                      # 40+ cognitive modules
│   ├── memory/                      # 6-layer memory system
│   ├── server/                      # HTTP API + web UI
│   ├── tools/                       # 51 built-in tools
│   └── training/                    # Training data collection
├── .github/workflows/               # CI + release automation
├── Dockerfile
├── docker-compose.yml
├── install.sh
└── go.mod                           # Zero external dependencies
```

---

## Documentation

| Document | What It Covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Deep dive into cognitive architecture and system design |
| [TESTING.md](TESTING.md) | Test methodology, benchmarks, coverage |
| [BENCHMARKS.md](BENCHMARKS.md) | Performance analysis across all operations |
| [SECURITY.md](SECURITY.md) | Security model, privacy guarantees, threat model |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome:

- New knowledge packages (any domain)
- Better training data for the neural classifier
- New tools and capabilities
- Improved generation strategies
- Bug fixes

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Aristotle** &mdash; for the name and the philosophy of active intellect
- **Mann & Thompson** &mdash; for Rhetorical Structure Theory
- **The Go team** &mdash; for a language that makes this possible in one binary

---

<p align="center">
  <strong>Nous thinks locally. No LLM required.</strong>
</p>
