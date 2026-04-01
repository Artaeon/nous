# Nous (νοῦς) — Architecture Reference

**Native Orchestration of Unified Streams**
Version 1.1.0 | Go 1.22+ | Zero External Dependencies | ~19MB Static Binary | 52 Built-in Tools | Mamba SSM

---

## Overview

Nous is a fully local AI assistant powered by a pure cognitive engine — no external models required. A deterministic NLU engine with 30+ intent categories and 52 built-in tools handles queries through pattern matching, knowledge graphs, and compositional generation. A cognitive compiler progressively compiles query patterns into deterministic handlers (~0ms), and a custom Mamba SSM with knowledge-constrained decoding generates neural responses with zero hallucination. The Thinking Engine handles remaining open-ended queries with frame-based generation and discourse planning, all running in ~50 MB RAM.

```
┌──────────────────────────────────────────────────────────────────┐
│                           USER INPUT                             │
│                    (Terminal / Telegram / Web UI / CLI pipes)     │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                 COGNITIVE COMPILER (first pass)                   │
│         Compiled pattern handlers, ~0ms                          │
│         Regex with named capture groups → slot-filled templates  │
│                                                                  │
│  HIT → execute compiled handler → done (~0ms)                    │
│  MISS ↓                                                          │
└──────┬───────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     NLU ENGINE (second pass)                      │
│         Deterministic intent classification, <1ms                 │
│         30+ intent categories, entity extraction                  │
│                                                                  │
│  Confidence ≥ 0.5 → ActionRouter → Direct tool dispatch (52)     │
│  weather, convert, timer, translate, volume, notes, todos,       │
│  calendar, hash, dict, network, process, app, brightness,        │
│  archive, diskusage, qrcode, screenshot, email, news, ...        │
│  → deterministic, <1ms-500ms                                     │
└──────┬───────────────────────────────────────────────────────────┘
       │ (NLU miss or low confidence)
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     QUERY CLASSIFICATION                         │
│              FastPathClassifier (51 regex patterns)               │
│                                                                  │
│  PathFast ──── greetings, yes/no, simple math, identity          │
│  PathMedium ── explanations, opinions, recall, introductions     │
│  PathFull ──── file ops, git, search, code, multi-step           │
└──────┬──────────────┬────────────────────────┬───────────────────┘
       │              │                        │
       ▼              ▼                        ▼
   FAST PATH      MEDIUM PATH              FULL PATH
   (0ms)          (0ms-10ms)               (10ms-500ms)
       │              │                        │
       ▼              ▼                        ▼
┌─────────────┐ ┌──────────────┐  ┌────────────────────────────┐
│ Canned      │ │ Response     │  │ Cognitive Engine           │
│ Greetings   │ │ Crystals     │  │                            │
│ (68 entries)│ │ (semantic    │  │ Tier 1: Intent Compiler    │
│             │ │  cache, 0ms) │  │   33 patterns → tool call  │
│ 0ms         │ │              │  │   → Response Synthesizer   │
│             │ │ HIT? → done  │  │   → 0ms, deterministic    │
│             │ │ MISS →       │  │                            │
│             │ │ Thinking     │  │ Tier 2: Thinking Engine    │
│             │ │ Engine       │  │   12 task types            │
└─────────────┘ └──────────────┘  │   Frame-based generation   │
                                  │   Discourse planning       │
                                  │                            │
                                  │ Tier 3: Full Reasoning     │
                                  │   Up to 6 tool iterations  │
                                  │   Knowledge graph lookup   │
                                  │   Compositional generation │
                                  └──────────┬─────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │ MAMBA SSM (neural generation) │
                              │ Constrained beam search       │
                              │ FactTrie → zero hallucination │
                              └──────────────────────────────┘
```

### Response Pipeline (linear summary)

```
User Query
    ↓
Cognitive Compiler (compiled handlers, ~0ms)
    ↓ (miss)
NLU Engine (neural + pattern, <1ms)
    ↓
ActionRouter (52 tools, knowledge graph)
    ↓
Mamba SSM (constrained neural generation)
    ↓
Composer (template + discourse planning)
    ↓
Quality Gate
    ↓
Response
```

---

## Core Architecture: 6 Cognitive Streams

Six goroutines running concurrently on a shared blackboard (pub/sub):

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Perceiver  │    │  Reasoner   │    │  Planner    │
│  (input     │───►│  (tool      │───►│  (goal      │
│   parsing)  │    │   calling)  │    │   decomp)   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
  ┌──────────────── BLACKBOARD ─────────────────┐
  │  Shared state, percepts, goals, answers     │
  └──────────────────────────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Executor   │    │  Reflector  │    │  Learner    │
│  (tool      │    │  (quality   │    │  (pattern   │
│   dispatch) │    │   eval)     │    │   mining)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

---

## Memory Architecture: 6 Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                       MEMORY SYSTEM                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Working Memory (64 slots)                                       │
│  ├── Relevance decay (0.01/sec)                                  │
│  ├── Built-in word embeddings for similarity search              │
│  └── In-memory only, per-session                                 │
│                                                                  │
│  Long-Term Memory (~/.nous/longterm.json)                        │
│  ├── Persistent key-value facts (user profile, project facts)    │
│  ├── Category-based retrieval                                    │
│  └── Auto-backed up (5 timestamped copies)                       │
│                                                                  │
│  Episodic Memory (.nous/episodes.json)                           │
│  ├── Every interaction recorded (10,000 cap)                     │
│  ├── Semantic search via built-in word embeddings                │
│  ├── Keyword search fallback                                     │
│  └── Success rate tracking per tool                              │
│                                                                  │
│  Project Memory (.nous/project_memory.json)                      │
│  ├── Per-project facts with confidence scores                    │
│  └── Contextual to working directory                             │
│                                                                  │
│  Knowledge Vector Store (.nous/knowledge.json)                   │
│  ├── 660 encyclopedic chunks (10 domains)                        │
│  ├── Semantic search via built-in 50-dimensional word vectors    │
│  └── Custom ingestion via /ingest command                        │
│                                                                  │
│  Response Crystal Store (.nous/response_crystals.json)           │
│  ├── 500-entry semantic cache of generated responses             │
│  ├── Learns from every cognitive generation (async, non-blocking)│
│  ├── 0.82 cosine similarity threshold for cache hits             │
│  └── Quality-weighted pruning (quality + recency + usage)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Speed Architecture: Progressive Compilation

The core insight: **the more you use Nous, the faster it gets.** Every generated response is "compiled" into a deterministic cache entry. Over weeks of use, the cache hit rate grows toward 95%.

### Response Path Hierarchy (fastest first)

| Layer | What it handles | Speed | Coverage |
|-------|----------------|-------|----------|
| 1. NLU + ActionRouter | 30+ intents: weather, convert, timer, translate, volume, notes, todos, etc. | **<1ms** | **40-60%** |
| 2. Canned greetings | "hello", "thanks", "bye" (68 entries) | **0ms** | 5-10% |
| 3. Intent compiler | "read file", "show commits" (33 patterns) | **200ms** | 10-20% |
| 4. Response synthesizer | Tool result formatting (8 tool types) | **instant** | (part of #3) |
| 5. Response crystals | Previously-answered questions (500 cache) | **28ms** | **grows over time** |
| 6. Phantom chain cache | Repeated tool chains (200 LRU, 60s TTL) | **instant** | varies |
| 7. Crystal recipes | Learned tool sequences (50 recipes) | **instant** | varies |
| 8. Speculative pre-computation | Follow-up predictions (20 LRU, 30s TTL) | **pre-loaded** | varies |
| 9. Thinking Engine | New open-ended questions (frame-based generation) | **10-100ms** | 10-15% |
| 10. Full cognitive pipeline | Complex/creative (up to 6 tool iterations) | **50-500ms** | 3-5% |

### Measured Performance

```
NLU instant:  "what's the weather?"  → NLU → tool → 67ms     (deterministic)
NLU instant:  "convert 5 km to mi"   → NLU → tool → <1ms     (deterministic)
NLU instant:  "set timer 5 min"      → NLU → tool → <1ms     (deterministic)
NLU instant:  "disk usage"           → NLU → tool → 8ms      (deterministic)
NLU instant:  "check network"        → NLU → tool → 637ms    (deterministic)
First time:   "what is relativity?"  → Thinking Engine → ~50ms
Second time:  "explain relativity"   → crystal hit → 28ms
Greetings:    "hello"                → canned → 0ms
Tool queries: "show recent commits"  → cognitive bypass → 200ms
```

### Projected Cache Hit Rate Over Time

```
Day 1:    80% instant (NLU handles 30+ intents + built-in patterns)
Week 1:   87% instant (50 response crystals learned)
Week 2:   91% instant (100 crystals + recipe patterns)
Month 1:  95% instant (200 crystals covering daily patterns)
Month 3:  98% instant (500 crystals, full vocabulary cached)
```

---

## Self-Improvement Architecture

Nous has 8 interconnected learning systems:

```
┌──────────────────────────────────────────────────────────────────┐
│                    SELF-IMPROVEMENT LOOP                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Response Crystallization                                     │
│     Every generated response → semantic cache → future instant   │
│     500 entries, 0.82 similarity threshold                       │
│                                                                  │
│  2. Training Pipeline                                            │
│     Every interaction → JSONL/Alpaca/ChatML training data        │
│     Optional: LoRA fine-tuning at 50+ quality pairs (advanced)   │
│     Quality scoring: success × speed × tool_usage                │
│                                                                  │
│  3. Self-Distillation (DPO)                                      │
│     Failures → contrastive pairs (chosen vs rejected)            │
│     System learns what NOT to do from its own mistakes           │
│                                                                  │
│  4. Neuroplastic Tool Descriptions                               │
│     Tool prompts evolve via genetic algorithm                    │
│     Success rate per description variant → best survive          │
│                                                                  │
│  5. Crystal Reasoning (Recipes)                                  │
│     Successful tool sequences → parameterized recipes            │
│     High-confidence recipes bypass generation entirely            │
│     Up to 50 recipes, confidence-weighted matching               │
│                                                                  │
│  6. Auto-Crystallization                                         │
│     Mines episodic memory for recurring patterns                 │
│     Generates instant-match crystals from frequent sequences     │
│                                                                  │
│  7. Neural Cortex (Policy Head)                                  │
│     Lightweight neural network predicting best tool for intent   │
│     L2 weight decay + learning rate decay prevents overfitting   │
│     Intent-Cortex ensemble: two systems vote together            │
│                                                                  │
│  8. Source Health Monitoring                                      │
│     Per-source quality EMA for virtual context                   │
│     Dynamic budget rebalancing — bad sources get less budget     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Anti-Hallucination Architecture: 6 Layers

```
Layer 1: Intent Compilation
  → Deterministic tool calls, zero hallucination possible

Layer 2: Neural Scaffolding
  → Pre-fill verified facts into response seed
  → Generator can only write connective tissue between facts

Layer 3: Embedding-Driven Grounding
  → Built-in 50-dimensional word vectors as semantic oracle
  → Tool/file matching by embedding similarity, deterministic

Layer 4: User Profile Grounding
  → "ONLY state facts listed here, never invent details"
  → Profile injected into system prompt

Layer 5: Post-Generation Validation
  → Check generated response against knowledge store
  → Flag contradictions before user sees them

Layer 6: Knowledge-Constrained Decoding (Mamba)
  → FactTrie restricts beam search to verified token sequences
  → Neural generation physically cannot produce ungrounded text
```

---

## Mamba Language Model (Selective State Spaces)

Nous includes a custom Mamba SSM implementation in pure Go — the first
decoder-only structured state space model built without any ML framework.

Architecture:
- Selective State Space Model (S6) with O(n) training, O(1)/token inference
- 8 Mamba blocks: conv1d → SiLU → input-dependent SSM → gated output
- 7.6M parameters (256-dim, 16-state, 4-conv, 2x expand)
- Stateful inference: hidden state updated per token, no KV cache
- Knowledge-constrained beam search: FactTrie ensures zero hallucination

Training: `nous-train mamba -knowledge knowledge/ -epochs 50`
Binary format: MAMB magic, auto-detected by Bridge loader

---

## Cognitive Compiler (Progressive Self-Improvement)

Every response generation feeds into the CognitiveCompiler, which extracts
query patterns and compiles them into deterministic handlers.

Pipeline:
1. Query → extractPattern → regex with named capture groups
2. Response → extractTemplate → slot-filled template
3. Future matching queries → execute compiled handler (~0ms)
4. Quality feedback loop → EMA-based scoring, auto-pruning

Over time: Day 1 = 80% instant, Week 1 = 90%, Month 1 = 99%

---

## Federated Crystal Sharing

Privacy-preserving collective intelligence:
- SharedCrystal: pattern + response template (no personal data)
- CrystalBundle: export/import with SHA-256 checksums
- Registry: file-based, search by keyword/intent
- TrustScorer: quality × votes × recency × bundle trust

REPL: `/federation status|export|import|search|top`

---

## UNIX CLI Primitives

Pipe-friendly cognitive infrastructure:
- `nous understand <text>` → JSON `{intent, action, entities, confidence}`
- `nous generate --facts "..." --style paragraph|bullet|brief`
- `nous reason <question>` → JSON `{question, analysis, conclusion}`
- `nous remember <key> [value]` → store/recall/list

All accept stdin, output JSON, composable with pipes.

---

## Multi-Hop Knowledge Reasoning

Traverses the knowledge graph to find connections between entities:
- Direct edges: A → B
- Two-hop paths: A → X → B
- Shared properties: common attributes
- Natural language explanation of connections

---

## Tool System: 52 Built-In Tools

### Core File Tools

| Tool | Function | Undo |
|------|----------|------|
| `read` | Read file contents with offset/limit | — |
| `write` | Create/overwrite files | ✓ |
| `edit` | Find-replace in files | ✓ |
| `patch` | Multi-line before/after edit | ✓ |
| `find_replace` | Regex find-replace | ✓ |
| `replace_all` | Bulk replace across files | — |
| `glob` | Find files by pattern | — |
| `grep` | Regex search in files | — |
| `ls` | List directory contents | — |
| `tree` | Directory tree (depth 3) | — |
| `mkdir` | Create directory tree | ✓ |
| `git` | Git operations (safe validated) | — |
| `diff` | Git diff / staged changes | — |
| `shell` | Shell execution (requires --trust) | — |
| `run` | Command execution (60s timeout) | — |
| `fetch` | Fetch URL content (1MB limit, 30s timeout) | — |
| `sysinfo` | OS, CPU, disk, hostname, IP | — |
| `clipboard` | Read/write system clipboard | — |

### Assistant Tools (Batch 1)

| Tool | Function |
|------|----------|
| `weather` | Weather via wttr.in (current, forecast) |
| `convert` | Unit conversion (length, weight, temperature, volume, speed, data) |
| `currency` | Currency exchange rates via frankfurter.app |
| `notes` | Create, list, search, delete notes |
| `todos` | Create, list, complete, delete todo items |
| `filefinder` | Smart file search with extension and directory awareness |
| `summarize` | Summarize URL content |
| `rss` | RSS/Atom feed reader |
| `coderunner` | Run code snippets (Python, Go, JavaScript, Bash) |
| `calendar` | Calendar events and date calculations |
| `email` | Check email via IMAP |
| `screenshot` | Take screenshots via grim/scrot |
| `websearch` | Web search via DuckDuckGo/SearXNG |

### Assistant Tools (Batch 2)

| Tool | Function |
|------|----------|
| `volume` | System volume control via pactl |
| `brightness` | Screen brightness via brightnessctl/sysfs |
| `notify` | Desktop notifications via notify-send |
| `timer` | Countdown timer with pomodoro support |
| `app` | App launcher, process finder/killer (.desktop files) |
| `hash` | Hash/encode: md5, sha256, base64, url, hex |
| `dict` | Dictionary definitions via dictionaryapi.dev |
| `netcheck` | Ping, DNS lookup, port check, connectivity test |
| `translate` | Language translation via Lingva API |
| `qrcode` | QR code generation/reading via qrencode/zbarimg |
| `archive` | Compress/extract tar.gz and zip archives |
| `diskusage` | Directory size analysis with top-N breakdown |
| `process` | Process list, search, kill, top (by memory/CPU) |

### NLU-Driven Tool Dispatch

30+ intent categories route queries directly to tools deterministically:

```
"what's the weather?"          → NLU → weather tool          → <100ms
"convert 10 miles to km"       → NLU → convert tool          → <1ms
"set a timer for 5 minutes"    → NLU → timer tool            → <1ms
"translate hello to spanish"   → NLU → translate tool         → <500ms
"turn up the volume"           → NLU → volume tool            → <100ms
"define serendipity"           → NLU → dict tool              → <500ms
"ping google.com"              → NLU → netcheck tool          → <1s
"hash this with sha256"        → NLU → hash tool              → <1ms
"show disk usage"              → NLU → diskusage tool         → <100ms
"open firefox"                 → NLU → app tool               → <100ms
"read main.go"                 → Exocortex → read(path=...)   → <200ms
"show recent commits"          → Exocortex → git(log...)      → <200ms
```

---

## Personalization Architecture

### First-Run Onboarding (5 questions)
```
? What's your name?
? What do you do?
? What are you working on? (optional)
? What are your interests? (optional)
? How should I help you? (optional)
```

### User Profile Injection
- All `user.*` facts from LTM injected into every generation context
- Cognitive engine knows user from the start of every conversation
- Anti-hallucination: "ONLY state facts listed here, never invent details"

### Fact Extraction
- Automatic pattern-based extraction from natural conversation
- Patterns: name, role, interests, current work, location
- Stored in LTM (persistent) + working memory (immediate)

### Welcome-Back
- Returning users see personalized greeting with name + last project
- Onboarding skipped when LTM has data

---

## Access Channels

```
┌─────────────────────────────────────────────────────────────┐
│                        NOUS SERVER                           │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Terminal  │  │ Web UI   │  │ Telegram │  │ Discord  │   │
│  │ (REPL)   │  │ (:3333)  │  │ (bot)    │  │ (bot)    │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │              │              │              │         │
│       └──────────────┴──────────────┴──────────────┘         │
│                          │                                   │
│                          ▼                                   │
│                 Same Cognitive Pipeline                       │
│                 Same Memory System                            │
│                 Same Learning Loop                            │
│                 Same Response Crystals                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

All channels go through identical processing:
1. Check cognitive compiler (compiled handlers)
2. Classify query (51 patterns)
3. Extract personal facts
4. Check response crystal cache
5. Route to appropriate path (fast/medium/full)
6. Record in episodic memory
7. Collect training data
8. Learn response crystal (async)

### Web UI (5 tabs)
- **Chat**: Terminal-style interface with task sidebar and job panel
- **Dashboard**: 8 metric cards (cognitive engine, memory, episodes, training, tasks)
- **Memory**: Tables for LTM, working memory, episodic history
- **Tools**: Tool catalog + clickable slash command reference
- **Settings**: Preferences, sessions, conversation history, connection

### HTTP API (20 endpoints)
```
POST /api/chat              — Send message, get response
POST /api/jobs              — Queue background task
GET  /api/jobs              — List jobs
GET  /api/jobs/{id}         — Inspect job
DELETE /api/jobs/{id}       — Cancel job
GET  /api/status            — System status
GET  /api/health            — Health check
GET  /api/dashboard         — Comprehensive system overview
GET  /api/memory            — Working memory items
GET  /api/longterm          — Long-term memory entries
GET  /api/episodes          — Episodic memory
GET  /api/tools             — Tool catalog
GET  /api/training          — Training data stats
GET  /api/sessions          — Saved session list
GET  /api/conversation      — Current conversation messages
GET  /api/assistant/today   — Notifications + schedule
GET  /api/assistant/tasks   — List/create tasks
POST /api/assistant/tasks/{id}/done — Mark completed
GET  /api/assistant/preferences     — Preferences
GET  /api/assistant/routines        — Routines
```

---

## Deployment Architecture

### Docker (recommended for servers)

```yaml
services:
  nous:
    build: .
    network_mode: host
    environment:
      - NOUS_TELEGRAM_TOKEN=...
    command: ["--serve", "--port", "3333",
              "--trust", "--api-key", "${NOUS_API_KEY}"]
    volumes:
      - nous_data:/data
```

- **Image size**: ~46MB (Alpine + 19MB binary + knowledge files)
- **Startup time**: ~2 seconds
- **Knowledge**: 660 chunks auto-ingested on first run (background)
- **Memory**: All backed up (5 timestamped copies per file)

### Resource Requirements

| Component | RAM | Disk | CPU |
|-----------|-----|------|-----|
| Nous binary | ~50 MB | 19 MB | negligible |
| Memory files | negligible | < 50 MB | — |
| **Total** | **~50 MB** | **~69 MB** | **any** |

No external services, no model downloads, no special hardware required.

### Security

- API key authentication (Bearer token) on all endpoints
- Web UI prompts for key, stores in localStorage
- Telegram allowlist (only specified user IDs)
- CORS restricted to localhost
- Atomic file writes with backups
- No secrets in Docker image
- HTTPS via Traefik/nginx reverse proxy

---

## Data Persistence

All data stored as plain JSON with atomic writes and automatic backups:

| File | Location | Backups | Content |
|------|----------|---------|---------|
| `longterm.json` | `~/.nous/` | 5 copies | User profile, facts |
| `episodes.json` | `.nous/` | 5 copies | Every interaction |
| `assistant.json` | `~/.nous/` | 5 copies | Tasks, routines, preferences |
| `sessions/*.json` | `~/.nous/sessions/` | 5 copies | Conversation history |
| `training_data.json` | `.nous/` | 5 copies | Fine-tuning data |
| `knowledge.json` | `.nous/` | — | Knowledge vector store |
| `response_crystals.json` | `.nous/` | 3 copies | Learned response cache |

---

## What Makes Nous Different

1. **Pure cognitive engine**: No external models required. A custom Mamba SSM (7.6M params, pure Go) handles neural generation, while the Thinking Engine, Discourse Planner, and Compositional Generation handle all other open-ended queries deterministically.

2. **Compiler-first architecture**: Every query first hits the Cognitive Compiler (~0ms compiled handlers), then the NLU engine (<1ms). Pattern matching, word lists, and entity extraction route to tools directly. Only queries neither can handle fall through to the cognitive pipeline.

3. **Progressive compilation**: Every generated response feeds back into the Cognitive Compiler and crystal cache. The system gets faster over time — from 80% instant on day 1 to 99% after a month.

4. **Zero dependencies**: Single Go binary, no Python, no npm, no Docker required for the binary itself. 52 tools implemented in pure Go + standard Linux utilities.

5. **Self-improving**: 8 learning systems evolve tool descriptions, cache patterns, collect training data, and learn from failures.

6. **~50 MB total**: Runs on any hardware — no downloads, no special hardware, instant startup.

7. **True personalization**: Onboarding → LTM → context injection → anti-hallucination grounding. Knows your name, interests, and work across sessions.

8. **Multi-channel, single brain**: Terminal, Web UI, Telegram, Discord, Matrix — all access the same NLU engine, memory, learning, and crystals.

---

*Built by Artaeon (raphael.lugmayr@stoicera.com) | MIT License*
