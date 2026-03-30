<p align="center">
  <img src="assets/banner.svg" alt="Nous" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white" alt="Go 1.22+">
  <img src="https://img.shields.io/badge/binary-16_MB-155e75?style=flat-square" alt="16 MB binary">
  <img src="https://img.shields.io/badge/RAM-~50_MB-0891b2?style=flat-square" alt="~50 MB RAM">
  <img src="https://img.shields.io/badge/dependencies-zero-16a34a?style=flat-square" alt="Zero dependencies">
  <img src="https://img.shields.io/badge/tools-52-f59e0b?style=flat-square" alt="52 tools">
  <img src="https://img.shields.io/badge/license-MIT-65a30d?style=flat-square" alt="MIT License">
</p>

---

Nous is a **fully local autonomous AI agent** that runs entirely on your machine. No LLM, no cloud, no API keys, no subscriptions. One static Go binary, 50 MB of RAM, works offline on any hardware including a Raspberry Pi.

Give it a goal and it executes — researching, writing reports, running tools, asking you when it needs input. Or use it as a daily assistant with 52 built-in tools, 680+ knowledge topics, and Socratic coaching.

## What makes Nous different

- **Zero hallucination** — every fact is grounded in the knowledge graph. It says what it knows or honestly says it doesn't know.
- **Autonomous agent** — give it a goal, it decomposes it, executes with tools, adapts when results are thin, and produces structured reports.
- **52 tools built-in** — timer, weather, calculator, translate, password, notes, todos, habits, expenses, code runner, web search, and 40 more. All local, all instant.
- **Socratic coaching** — asks probing questions instead of giving generic advice. Knows when questions are more valuable than answers.
- **Self-improving** — learns your preferences, detects patterns in your conversations, tracks its own strengths and weaknesses.
- **< 50ms response time** — no API calls, no network latency for local queries.
- **~50 MB RAM** — runs on anything. No GPU required.
- **Pure Go, zero dependencies** — one `go build` and you're done.

## Quick start

```bash
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous
./nous
```

No `pip install`. No model downloads. No GPU drivers. First launch trains the neural classifier (~90 seconds). After that, it starts instantly.

### Server mode

```bash
./nous --serve --port 3333 --api-key yoursecretkey

# Chat
curl -X POST http://localhost:3333/api/chat \
  -H "Authorization: Bearer yoursecretkey" \
  -H "Content-Type: application/json" \
  -d '{"message": "what is quantum mechanics"}'

# Start an agent goal
curl -X POST http://localhost:3333/api/agent/start \
  -H "Authorization: Bearer yoursecretkey" \
  -H "Content-Type: application/json" \
  -d '{"goal": "research artificial intelligence and write a summary"}'
```

## What it actually does

### Knowledge — 680+ topics, Wikipedia-quality paragraphs

```
nous > what is quantum mechanics
Quantum mechanics is the branch of physics describing the behavior
of matter and energy at atomic and subatomic scales. Developed in the
early twentieth century by Planck, Heisenberg, Schrodinger, and Dirac,
it replaces deterministic classical mechanics with probabilistic wave
functions. Key principles include the uncertainty principle and
superposition. It predicts phenomena such as quantum tunneling,
entanglement, and zero-point energy, forming the foundation for
modern technologies including transistors, MRI machines, and quantum
computers.
```

Every answer comes from real human-written text in the knowledge base. 5,000+ typed facts extracted from 78,000 words of curated knowledge.

### Autonomous agent — give it a goal, it executes

```
nous > /agent start "research machine learning and write a summary"

[PLANNING] Decomposed into 5 phases:
  Phase 1: Web research (3 searches)
  Phase 2: Synthesize findings
  Phase 3: Generate structured report
  Phase 4: Review and evaluate
  Phase 5: Final report

[EXECUTING] Phase 1 — searching DuckDuckGo...
[EXECUTING] Phase 2 — synthesizing with knowledge graph...
[EXECUTING] Phase 3 — generating report...

[COMPLETE] 11 tasks, 17 tool calls, 15 seconds
Report saved to ~/.nous/agent/research_report.md
```

The agent can:
- Research any topic (web search + knowledge graph)
- Write structured markdown reports (5 sections)
- Adapt when results are thin (retries with new queries)
- Pause for your input when it needs guidance
- Persist state across restarts
- Run on a schedule (daily, hourly, weekly)

### 52 tools — all local, all instant

| Category | Tools |
|---|---|
| **Information** | weather (live), dictionary, translate, websearch, summarize, realtime data |
| **Productivity** | notes, todos, calendar, email, timer, reminder |
| **Life** | journal, habits, expenses, bookmarks, passwords |
| **System** | sysinfo, clipboard, screenshot, volume, brightness, notify, processes |
| **Compute** | calculator, convert, currency, hash, qrcode, coderunner (Python/Bash/JS) |
| **Files** | read, write, edit, glob, grep, filefinder, archive, diskusage |
| **Network** | netcheck, fetch, websearch |

```
nous > set a timer for 25 minutes
Timer started: 25m0s (fires at 15:34)

nous > whats 25% of 480
0.25*480 = 120

nous > translate hello to french
hello -> bonjour (French)

nous > whats the weather in Vienna
Weather in Vienna: partly cloudy, 12.5C, humidity 48%, wind 7.2 km/h

nous > generate a strong password
Password: kR7#mQ2x$Lp9!wBn
```

### Socratic coaching — asks better questions than it gives answers

```
nous > should I quit my job to start a business

Before I give you my take, let me help you clarify what matters most:

1. What prompted your interest in this career decision at this moment?
2. How would this decision look if you couldn't reverse it?
3. What evidence supports this direction?
```

Nous detects when you're making a decision, exploring an idea, or working through a problem — and asks probing questions instead of giving generic advice. Five coaching modes: Explore, Decide, Coach, Challenge, Deepen.

### Format compliance — follows instructions exactly

```
nous > summarize stoicism in 3 bullet points

- Stoicism was a Hellenistic philosophical school founded by Zeno
  of Citium around 300 BCE in Athens.
- They held that the universe is governed by a rational principle
  called the Logos, and that living in accordance with nature and
  reason constitutes the path to tranquility.
- Stoic practices of emotional regulation, acceptance of fate, and
  focus on what lies within one's control have experienced a
  significant revival in modern self-help and cognitive behavioral
  therapy.
```

Ask for 3 bullets, get exactly 3. Ask for one sentence, get one sentence. Ask for 3 questions, get exactly 3.

### Emotional intelligence

```
nous > i had a terrible day
That must weigh on you. I wish I could do more.
Would it help to talk it through?

nous > i just got promoted!
That's amazing! I love hearing that.
```

Nous matches emotional tone — warm for bad news, celebratory for good news, practical for requests.

## Cognitive architecture

Nous isn't an LLM wrapper. It's a cognitive architecture built from scratch:

```
User Query
    |
    v
NLU (neural classifier, 12K examples, 98% accuracy)
    |
    v
ActionRouter --- 52 tools, knowledge graph, reasoning engines
    |
    v
Response Generation --- hybrid NLG, Wikipedia paragraphs, format compliance
    |
    v
Quality Gate --- filters contamination, filler, parroting
    |
    v
Response
```

### Innovation systems (unique to Nous)

| System | What it does |
|---|---|
| **Socratic Engine** | Detects when to ask instead of answer. 5 modes, 79 question templates. |
| **Insight Crystallizer** | Finds patterns across conversations — recurring themes, tensions, blind spots. |
| **Self-Model** | Tracks its own performance per domain. Knows what it's good and bad at. |
| **Knowledge Synthesis** | Reasons from adjacent knowledge instead of saying "I don't know." |
| **Cognitive Transparency** | Shows its reasoning: what was retrieved, what was inferred, what's uncertain. |
| **Deep Reasoner** | Multi-step reasoning chains: decompose question, answer parts, chain results. |

### Memory system — 6 layers, persistent

| Layer | What it stores |
|---|---|
| **Working** | Current conversation context (64 slots) |
| **Long-term** | Personal facts, preferences |
| **Episodic** | Every interaction, timestamped and searchable |
| **Project** | Per-directory project context |
| **Knowledge** | 5,000+ facts + 680 Wikipedia paragraphs |
| **Growth** | Interest tracking, learned patterns |

## Real-world use cases

### Personal productivity
- Morning briefing: weather, habits, todos, expenses, schedule
- Task and habit tracking with streaks
- Expense logging and summaries
- Notes and bookmarks, all local
- Pomodoro timer for focused work

### Research and analysis
- "Research X and write a report" — autonomous execution
- Web search + knowledge graph synthesis
- Structured markdown reports with sections
- Scheduled monitoring ("check AI news weekly")

### Decision support
- Socratic coaching for career, business, personal decisions
- Pros/cons comparison with real data
- Pattern detection across conversations ("you've mentioned this 3 times")

### Developer toolkit
- Run Python, Bash, JavaScript directly
- File operations, grep, diff, git
- System monitoring, process management
- Password generation, hash computation

### Privacy-first alternative
- Zero data leaves your machine
- No API keys or cloud accounts
- All data stored as local JSON/plain text files
- Full control over your information

## Agent commands

| Command | What it does |
|---|---|
| `/agent start "goal"` | Start autonomous execution of a goal |
| `/agent status` | Check current progress |
| `/agent report` | Get detailed progress report |
| `/agent input "response"` | Provide human input when agent pauses |
| `/agent stop` | Stop execution (can resume later) |
| `/agent schedule "goal" "daily 9:00"` | Schedule recurring goals |
| `/agent jobs` | List scheduled jobs |

## Chat commands

| Command | What it does |
|---|---|
| `/briefing` | Morning briefing — weather, tasks, habits |
| `/remind <when> <what>` | Set a reminder |
| `/remember <key> <value>` | Store a personal fact |
| `/recall <query>` | Search all memory layers |
| `/plan <goal>` | Generate a step-by-step plan |
| `/todos` | Show task list |
| `/habits` | Show habit tracking |
| `/journal` | Write a journal entry |
| `/tools` | List all 52 tools |
| `/version` | Version info |

## Honest comparison

|  | Cloud LLMs | Local LLMs (Ollama) | Nous |
|---|---|---|---|
| **Runs on** | Cloud servers | Your GPU (4-16 GB) | **Any CPU (50 MB)** |
| **Latency** | 500ms-3s | 100ms+ | **< 50ms** |
| **Privacy** | Data leaves your machine | Local | **100% local** |
| **Dependencies** | API key + internet | Python + model download | **Zero** |
| **Memory** | Per-session only | None | **Persistent, 6 layers** |
| **Tools** | Needs plugins | Needs wrappers | **52 built-in** |
| **Autonomous agent** | Limited | Limited | **Full (goal → plan → execute → report)** |
| **Hallucination** | 5-15% | 5-10% | **0% (grounded)** |
| **Conversation quality** | Excellent | Good | Basic |
| **Creative writing** | Excellent | Good | Basic |
| **Cost** | $20/month+ | Free (hardware) | **Free** |

## Architecture

```
nous/
├── cmd/
│   ├── nous/           # Main binary — REPL + HTTP server + agent
│   ├── nous-train/     # Offline neural model training
│   └── wikiimport/     # Wikipedia → knowledge packages
├── internal/
│   ├── agent/          # Autonomous agent — planning, execution, scheduling
│   ├── cognitive/      # NLU, NLG, knowledge graph, reasoning, coaching
│   ├── memory/         # 6-layer persistent memory
│   ├── tools/          # 52 built-in tools
│   ├── eval/           # Quality evaluation, red-team suites, KPIs
│   ├── training/       # Distillation, preference optimization
│   ├── server/         # HTTP API
│   └── hands/          # Multi-step task automation
├── knowledge/          # Wikipedia-quality text (78K words, 680+ topics)
├── packages/           # Structured knowledge packages
└── go.mod              # Zero external dependencies
```

## Requirements

- Go 1.22+ (build only)
- ~50 MB RAM
- No GPU
- No internet (after first build — weather/search need network)
- Linux, macOS, or Windows

## License

MIT. See [LICENSE](LICENSE).

---

<p align="center">
  Built by <a href="https://github.com/artaeon">Artaeon</a>
</p>
