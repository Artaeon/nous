# Nous (νοῦς) — Architecture Reference

**Native Orchestration of Unified Streams**
Version 0.9.0 | Go 1.22+ | Zero External Dependencies | ~14MB Static Binary | 45 Built-in Tools

---

## Overview

Nous is a fully local AI assistant that treats the LLM as a peripheral device, not the brain. A deterministic NLU engine with 30+ intent categories and 45 built-in tools handles 70-95% of queries without any LLM call. The system gets faster the more you use it — every LLM response teaches Nous to answer that type of question instantly next time.

```
┌──────────────────────────────────────────────────────────────────┐
│                           USER INPUT                             │
│                    (Terminal / Telegram / Web UI)                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     NLU ENGINE (first pass)                       │
│         Deterministic intent classification, <1ms                 │
│         30+ intent categories, entity extraction                  │
│                                                                  │
│  Confidence ≥ 0.5 → ActionRouter → Direct tool dispatch          │
│  weather, convert, timer, translate, volume, notes, todos,       │
│  calendar, hash, dict, network, process, app, brightness,        │
│  archive, diskusage, qrcode, screenshot, email, news, ...        │
│  → 0 LLM calls, <1ms-500ms                                      │
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
   (0-5s)         (0ms-10s)                (200ms-40s)
       │              │                        │
       ▼              ▼                        ▼
┌─────────────┐ ┌──────────────┐  ┌────────────────────────────┐
│ Canned      │ │ Response     │  │ Exocortex 3-Tier Engine    │
│ Greetings   │ │ Crystals     │  │                            │
│ (68 entries)│ │ (semantic    │  │ Tier 1: Intent Compiler    │
│             │ │  cache, 0ms) │  │   33 patterns → tool call  │
│ 0ms         │ │              │  │   → Response Synthesizer   │
│             │ │ HIT? → done  │  │   → 0ms, no LLM           │
│             │ │ MISS → LLM   │  │                            │
│             │ │ → Learn      │  │ Tier 2: Neural Scaffold    │
└─────────────┘ └──────────────┘  │   Facts pre-filled         │
                                  │   LLM fills gaps only      │
                                  │   → 1 LLM call             │
                                  │                            │
                                  │ Tier 3: Full Reasoning     │
                                  │   Up to 6 tool iterations  │
                                  │   Native tool calling API  │
                                  │   → 1-7 LLM calls          │
                                  └────────────────────────────┘
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
│  ├── Semantic embeddings for similarity search                   │
│  └── In-memory only, per-session                                 │
│                                                                  │
│  Long-Term Memory (~/.nous/longterm.json)                        │
│  ├── Persistent key-value facts (user profile, project facts)    │
│  ├── Category-based retrieval                                    │
│  └── Auto-backed up (5 timestamped copies)                       │
│                                                                  │
│  Episodic Memory (.nous/episodes.json)                           │
│  ├── Every interaction recorded (10,000 cap)                     │
│  ├── Semantic search via embeddings (nomic-embed-text)           │
│  ├── Keyword search fallback                                     │
│  └── Success rate tracking per tool                              │
│                                                                  │
│  Project Memory (.nous/project_memory.json)                      │
│  ├── Per-project facts with confidence scores                    │
│  └── Contextual to working directory                             │
│                                                                  │
│  Knowledge Vector Store (.nous/knowledge.json)                   │
│  ├── 660 encyclopedic chunks (10 domains)                        │
│  ├── Semantic search via nomic-embed-text embeddings             │
│  └── Custom ingestion via /ingest command                        │
│                                                                  │
│  Response Crystal Store (.nous/response_crystals.json)           │
│  ├── 500-entry semantic cache of LLM responses                   │
│  ├── Learns from every LLM call (async, non-blocking)            │
│  ├── 0.82 cosine similarity threshold for cache hits             │
│  └── Quality-weighted pruning (quality + recency + usage)        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Speed Architecture: Progressive Compilation

The core insight: **the more you use Nous, the faster it gets.** Every LLM response is "compiled" into a deterministic cache entry. Over weeks of use, the LLM bypass rate grows from 65% toward 95%.

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
| 9. Fast-path LLM | New simple questions | **3-10s** | 5-10% |
| 10. Full pipeline LLM | Complex/creative (up to 6 tool iterations) | **8-40s** | 3-5% |

### Measured Performance

```
NLU instant:  "what's the weather?"  → NLU → tool → 67ms     (0 LLM calls)
NLU instant:  "convert 5 km to mi"   → NLU → tool → <1ms     (0 LLM calls)
NLU instant:  "set timer 5 min"      → NLU → tool → <1ms     (0 LLM calls)
NLU instant:  "disk usage"           → NLU → tool → 8ms      (0 LLM calls)
NLU instant:  "check network"        → NLU → tool → 637ms    (0 LLM calls)
First time:   "what is relativity?"  → LLM → 12 seconds
Second time:  "explain relativity"   → crystal hit → 28 milliseconds (464x faster)
Greetings:    "hello"                → canned → 0ms
Tool queries: "show recent commits"  → exo-bypass → 200ms
```

### Projected LLM Bypass Rate Over Time

```
Day 1:    80% bypass (NLU handles 30+ intents + built-in patterns)
Week 1:   87% bypass (50 response crystals learned)
Week 2:   91% bypass (100 crystals + recipe patterns)
Month 1:  95% bypass (200 crystals covering daily patterns)
Month 3:  98% bypass (500 crystals, full vocabulary cached)
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
│     Every LLM response → semantic cache → future instant hits    │
│     500 entries, 0.82 similarity threshold                       │
│                                                                  │
│  2. Training Pipeline                                            │
│     Every interaction → JSONL/Alpaca/ChatML training data        │
│     Auto-triggers LoRA fine-tuning at 50+ quality pairs          │
│     Quality scoring: success × speed × tool_usage                │
│                                                                  │
│  3. Self-Distillation (DPO)                                      │
│     Failures → contrastive pairs (chosen vs rejected)            │
│     Model learns what NOT to do from its own mistakes            │
│                                                                  │
│  4. Neuroplastic Tool Descriptions                               │
│     Tool prompts evolve via genetic algorithm                    │
│     Success rate per description variant → best survive          │
│                                                                  │
│  5. Crystal Reasoning (Recipes)                                  │
│     Successful tool sequences → parameterized recipes            │
│     High-confidence recipes bypass LLM entirely                  │
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

## Anti-Hallucination Architecture: 5 Layers

```
Layer 1: Intent Compilation
  → Deterministic tool calls, zero hallucination possible

Layer 2: Neural Scaffolding
  → Pre-fill verified facts into response seed
  → LLM can only write connective tissue between facts

Layer 3: Embedding-Driven Grounding
  → nomic-embed-text as semantic oracle
  → Tool/file matching by embedding similarity, not LLM guess

Layer 4: User Profile Grounding
  → "ONLY state facts listed here, never invent details"
  → Profile injected into system prompt

Layer 5: Post-Generation Validation
  → Check LLM response against knowledge store
  → Flag contradictions before user sees them
```

---

## Tool System: 45 Built-In Tools

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

30+ intent categories route queries directly to tools without LLM:

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
- All `user.*` facts from LTM injected into every system prompt
- Model knows user from first token of every conversation
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
1. Classify query (51 patterns)
2. Extract personal facts
3. Check response crystal cache
4. Route to appropriate path (fast/medium/full)
5. Record in episodic memory
6. Collect training data
7. Learn response crystal (async)

### Web UI (5 tabs)
- **Chat**: Terminal-style interface with task sidebar and job panel
- **Dashboard**: 8 metric cards (model, memory, episodes, training, tasks)
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
    command: ["--serve", "--port", "3333", "--model", "qwen3:4b",
              "--trust", "--api-key", "${NOUS_API_KEY}"]
    volumes:
      - nous_data:/data
```

- **Image size**: 41MB (Alpine + 11MB binary + knowledge files)
- **Startup time**: ~8 seconds
- **Knowledge**: 660 chunks auto-ingested on first run (background)
- **Memory**: All backed up (5 timestamped copies per file)

### Resource Requirements

| Component | RAM | Disk | CPU |
|-----------|-----|------|-----|
| Nous binary | 30 MB | 11 MB | negligible |
| Ollama + qwen3:4b | 3 GB | 2.5 GB | moderate |
| Ollama + nomic-embed-text | 500 MB | 300 MB | light |
| Memory files | negligible | < 50 MB | — |
| **Total** | **~3.5 GB** | **~3 GB** | **2+ cores** |

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

1. **LLM-as-peripheral**: Deterministic code is the brain. A rule-based NLU engine with 30+ intent categories and 45 tools handles most queries without any LLM call. The LLM is only invoked when reasoning is genuinely needed.

2. **NLU-first architecture**: Every query hits the NLU engine first (<1ms). Pattern matching, word lists, and entity extraction route to tools directly. Only queries the NLU can't handle fall through to the LLM pipeline.

3. **Progressive compilation**: Every LLM response becomes a cached crystal. The system gets faster over time — from 80% bypass on day 1 to 98% after months.

4. **Zero dependencies**: Single Go binary, no Python, no npm, no Docker required for the binary itself. 45 tools implemented in pure Go + standard Linux utilities.

5. **Self-improving**: 8 learning systems evolve tool descriptions, cache patterns, collect training data, fine-tune models, and learn from failures.

6. **Tiny model effectiveness**: Makes qwen3:4b (2.5GB) genuinely useful through NLU bypass, scaffolding, grounding, and caching.

7. **True personalization**: Onboarding → LTM → system prompt injection → anti-hallucination grounding. Knows your name, interests, and work across sessions.

8. **Multi-channel, single brain**: Terminal, Web UI, Telegram, Discord, Matrix — all access the same NLU engine, memory, learning, and crystals.

---

*Built by Artaeon (raphael.lugmayr@stoicera.com) | MIT License*
