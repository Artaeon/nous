<h1 align="center">
  🧠 Nous
</h1>

<p align="center">
  <strong>Native Orchestration of Unified Streams</strong>
</p>

<p align="center">
  <em>A cognitive coding agent that runs entirely on your machine.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat-square&logo=go&logoColor=white" alt="Go 1.22+">
  <img src="https://img.shields.io/badge/binary-~8_MB-blue?style=flat-square" alt="~8 MB binary">
  <img src="https://img.shields.io/badge/GPU-not_required-green?style=flat-square" alt="No GPU">
  <img src="https://img.shields.io/badge/cloud-not_required-green?style=flat-square" alt="No Cloud">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen?style=flat-square" alt="MIT License">
</p>

---

> *"It is the active intellect that makes all things."*
> — Aristotle, *De Anima*, on nous (vouc)

---

## What Is This

Nous is an autonomous AI coding agent — like Claude Code or Cursor — but it runs **entirely on your local hardware** via Ollama. No cloud. No API keys. No telemetry. No data leaving your machine. It is built as a concurrent cognitive architecture: six independent processing streams (perceive, reason, plan, execute, reflect, learn) communicate through a shared blackboard, producing emergent intelligent behavior from the interplay of simple, well-defined modules. The result is a self-contained ~8 MB Go binary that can read your codebase, reason about it, write files, execute commands, and learn from its own experience — all running on CPU with whatever model you point it at.

---

## Feature Highlights

| Capability | Cloud AI Agents | Nous |
|---|---|---|
| **Data privacy** | Your code hits external servers | Everything stays on your machine |
| **API keys / billing** | Required | None |
| **Internet required** | Yes | No (after model download) |
| **Binary size** | Electron app / cloud service | ~8 MB static binary |
| **GPU required** | Typically yes | No (CPU inference via Ollama) |
| **Model choice** | Vendor-locked | Any Ollama-compatible model |
| **Session persistence** | Cloud-dependent | Local JSON, fully portable |
| **Tool confirmation** | Varies | Explicit user approval for destructive actions |
| **Self-improvement** | No | Learns patterns from successful interactions |
| **Architecture** | Monolithic LLM call | 6 concurrent cognitive streams |
| **Cost** | $20+/month | Free forever |

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              B L A C K B O A R D            │
                        │                                             │
                        │  Percepts ─ Goals ─ Plans ─ Actions ─ Memory│
                        │          (shared cognitive workspace)       │
                        └──────┬──────┬──────┬──────┬──────┬──────┬───┘
                               │      │      │      │      │      │
                 ┌─────────────┘      │      │      │      │      └─────────────┐
                 │        ┌───────────┘      │      │      └───────────┐        │
                 │        │        ┌─────────┘      └─────────┐        │        │
                 ▼        ▼        ▼                          ▼        ▼        ▼
          ┌──────────┐┌──────────┐┌──────────┐        ┌──────────┐┌──────────┐┌──────────┐
          │PERCEIVER ││ REASONER ││ PLANNER  │        │ EXECUTOR ││REFLECTOR ││ LEARNER  │
          │          ││          ││          │        │          ││          ││          │
          │ Parses   ││ Chain-of ││ Decom-   │        │ Runs     ││ Monitors ││ Extracts │
          │ input,   ││ -thought ││ poses    │        │ tools,   ││ quality, ││ patterns │
          │ extracts ││ with     ││ goals    │        │ records  ││ flags    ││ from     │
          │ intent & ││ autonomo ││ into     │        │ results  ││ errors,  ││ success, │
          │ entities ││ us tool  ││ ordered  │        │ on the   ││ triggers ││ persists │
          │          ││ use      ││ steps    │        │ board    ││ replans  ││ to disk  │
          └──────────┘└──────────┘└──────────┘        └──────────┘└──────────┘└──────────┘
                 │         │          │                     │          │           │
                 └─────────┴──────────┴─────────┬──────────┴──────────┴───────────┘
                                                │
                                         ┌──────┴──────┐
                                         │   Ollama    │
                                         │  (local)    │
                                         └─────────────┘
```

All six streams run as independent goroutines. They do not call each other — they communicate exclusively through the blackboard. This means the system can perceive new input while simultaneously executing a plan step, reflecting on a previous result, and learning from a completed goal.

---

## Getting Started

### Prerequisites

- **Go 1.22+** — [golang.org/dl](https://golang.org/dl/)
- **Ollama** — [ollama.com](https://ollama.com/)
- A model pulled locally (see [Recommended Models](#recommended-models))

### Install

```bash
# Clone
git clone https://github.com/artaeon/nous.git
cd nous

# Build (~8 MB binary, zero runtime dependencies)
make build

# Or install directly
go install github.com/artaeon/nous/cmd/nous@latest
```

### Run

```bash
# Basic — uses qwen2.5:1.5b by default
./nous

# With a specific model
./nous --model codellama:7b

# Enable shell command execution
./nous --allow-shell

# Full autonomy (shell + skip confirmations)
./nous --allow-shell --trust

# Resume a previous session
./nous --resume 1709834521000
```

### First Interaction

```
                ╔═══════════════════════════════════╗
                ║             N O U S               ║
                ║   Native Orchestration of         ║
                ║       Unified Streams             ║
                ╚═══════════════════════════════════╝

  version 0.5.0 | amd64 | 8 cores | 16 GB RAM

  connecting to ollama... OK (qwen2.5:1.5b)
  scanning project... nous (Go, 28 files)
  project memory: 0 facts
  6 cognitive streams active
  18 tools: read, write, edit, glob, grep, ls, shell, mkdir, tree, ...
  session: 1709834521000

  I am Nous. I think, therefore I am — locally.
  type /help for commands, /quit to exit

  nous> What does this project do?
```

---

## Configuration

All configuration is via CLI flags. No config files, no environment variables, no hidden state.

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen2.5:1.5b` | Ollama model to use |
| `--host` | `http://localhost:11434` | Ollama server address |
| `--memory` | `~/.nous` | Path for persistent memory and session storage |
| `--allow-shell` | `false` | Enable shell command execution |
| `--trust` | `false` | Skip confirmation prompts for file and shell operations |
| `--resume` | (none) | Resume a previous session by ID |
| `--version` | — | Print version and exit |

---

## Tools

Nous ships with 18 built-in tools that the reasoning engine can invoke autonomously. The model decides which tools to call and in what order — chaining multiple calls in a loop until it has enough information to answer.

| Tool | Description | Requires `--allow-shell` |
|---|---|---|
| `read` | Read file contents with optional line offset and limit | No |
| `write` | Create or overwrite a file (supports undo) | No (prompts for confirmation) |
| `edit` | Replace a specific string in a file (exact match, must be unique, supports undo) | No (prompts for confirmation) |
| `glob` | Find files matching a glob pattern (e.g. `*.go`) | No |
| `grep` | Search file contents for a regex pattern with optional file filter | No |
| `ls` | List directory contents with file sizes | No |
| `tree` | Show project directory structure with configurable depth | No |
| `mkdir` | Create a directory and all parent directories (supports undo) | No (prompts for confirmation) |
| `shell` | Execute an arbitrary shell command | **Yes** |
| `fetch` | HTTP GET with HTML tag stripping (1MB limit) | No |
| `run` | Execute a command with stdin support and 60s timeout | **Yes** |
| `sysinfo` | Show OS, architecture, CPU, disk space | No |
| `clipboard` | Read from or write to the system clipboard (via xclip/xsel) | No |
| `find_replace` | Regex find-and-replace in a file (supports undo) | No (prompts for confirmation) |
| `git` | Run git commands (status, diff, log, add, commit, etc.) | No |
| `patch` | Multi-line before/after text replacement | No (prompts for confirmation) |
| `replace_all` | Replace all occurrences of a string across multiple files | No (prompts for confirmation) |
| `diff` | Show git diff (working directory or staged) | No |

Destructive tools (`write`, `edit`, `shell`, `mkdir`, `patch`, `replace_all`, `find_replace`) require explicit user confirmation unless `--trust` is set. File modifications are tracked on an undo stack — use `/undo` to revert.

---

## Slash Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/status` | Show cognitive system status (percepts, goals, memory, conversation) |
| `/memory` | Show working memory contents ranked by relevance |
| `/longterm` | Show long-term memory entries with access counts |
| `/remember <key> <value>` | Store a project fact (persists across sessions) |
| `/recall <query>` | Search project memory by keyword |
| `/forget <key>` | Remove a project fact |
| `/undo` | Revert the last file change |
| `/history` | Show the undo stack |
| `/goals` | Show active goals and their status |
| `/model` | Show current model info and list all available Ollama models |
| `/tools` | List all registered tools with descriptions |
| `/project` | Show detected project info (language, file count, key files, tree) |
| `/sessions` | List all saved sessions with message counts |
| `/save [name]` | Save the current session with an optional name |
| `/clear` | Clear conversation context |
| `/quit` | Save session and exit |

---

## How the Cognitive Architecture Works

Classical AI agents make a single LLM call per user message. Nous works differently.

**Six concurrent streams** run as goroutines, communicating through a shared **blackboard** — a thread-safe cognitive workspace that holds percepts, goals, plans, action records, and working memory. No stream calls another directly. They react to events:

1. **Perceiver** — Listens for raw user input. Extracts intent and entities via the LLM. Posts a structured `Percept` to the blackboard.

2. **Reasoner** — Reacts to percepts. Runs autonomous chain-of-thought inference with tool use. Chains up to 8 tool calls in a single loop — reading files, searching code, editing files — before producing a final answer. Streams tokens to the terminal in real time, suppressing tool-call JSON from the display. Protected by the **Cognitive Grounding** system (see below).

3. **Planner** — Reacts to new goals. Decomposes them into ordered, executable step sequences using hierarchical task decomposition via the LLM.

4. **Executor** — Reacts to plans. Walks through each step, runs the specified tool, and records success or failure as an `ActionRecord` on the blackboard.

5. **Reflector** — Reacts to action records. Evaluates output quality via the LLM. If an action failed or produced questionable results, flags it for re-planning.

6. **Learner** — Reacts to completed goals. Extracts behavioral patterns from successful interactions and persists them to disk as JSON. Over time, the system builds a library of proven strategies.

**Memory is triple-layered:**

- **Working Memory** — Capacity-limited (64 slots), decay-based. Items lose relevance over time and are evicted when capacity is exceeded. Accessing an item boosts its relevance (recency effect).
- **Long-Term Memory** — Persistent JSON-backed key-value store with access counting. Survives restarts. Categories enable structured retrieval.
- **Project Memory** — Per-project fact store (`.nous/project_memory.json`). Stores conventions, architecture decisions, and key patterns. Accessible via `/remember`, `/recall`, `/forget`.

**Undo Stack** — All file-modifying tools (write, edit, find_replace, mkdir) push entries onto an undo stack. Use `/undo` to revert the most recent change, `/history` to view the stack.

**Context Compression** — The `compress` module distills conversation fragments into dense "atoms" — reusable `{trigger, knowledge, weight}` tuples. Automatically triggered when context budget exceeds 75%.

**Cognitive Grounding** (v0.5.0) — A five-layer system that prevents hallucinations in small models:

1. **Progressive Tool Disclosure** — Instead of injecting all 18 tools into the system prompt (~1000 tokens), only the 5-8 tools relevant to the detected intent are shown. The model can request additional tool categories via a `request_tools` meta-tool. Saves ~500 tokens of context.

2. **Smart Result Truncation** — Tool-specific truncation keeps results compact: file reads show first/last 20 lines, grep/glob cap at 15 matches, directory listings cap at 30 entries. Universal 2048-char hard limit.

3. **Result Validation** — Every tool result is checked for common issues (empty reads, missing files, permission errors) and annotated with corrective hints for the model.

4. **Context Budget Tracking** — Estimates token consumption across all messages. Auto-compresses old conversation turns at 75% usage (via atoms or rule-based fallback). Forces a final answer at 85% to prevent context overflow.

5. **Synchronous Reflection Gate** — After each tool call, a rule-based validator checks for: errors, empty results, repeated calls, and excessive iterations. Injects corrective hints and forces the model to answer when it's stuck in a loop.

**Cognitive Pipeline** (v0.5.0) — Instead of accumulating messages that fill the context window and cause degradation after 3-4 steps, each reasoning step gets a fresh conversation with only compressed one-line summaries of previous steps. At step 5, context usage is ~15% instead of ~80%, enabling 8+ tool calls without quality degradation.

**Multi-Model Router** (v0.5.0) — Auto-discovers available Ollama models at startup and routes cognitive tasks to the best model: perception → tinyllama (~200ms), reasoning → qwen2.5:1.5b, compression → tinyllama. Falls back to a single model when only one is available.

**Codebase Index** (v0.5.0) — At startup, parses all Go files using `go/parser` and `go/ast` to extract function signatures, types, interfaces, and methods. Stored in `.nous/index.json`. The reasoner injects precise structural context in ~50 tokens instead of reading 500 lines. Supports incremental updates via file hash comparison.

---

## Project Structure

```
nous/
├── cmd/
│   └── nous/
│       └── main.go              # Entry point, REPL, CLI flags, signal handling
├── internal/
│   ├── blackboard/
│   │   └── blackboard.go        # Shared cognitive workspace + event bus
│   ├── cognitive/
│   │   ├── stream.go            # Stream interface + Base struct
│   │   ├── perceiver.go         # Input parsing + intent extraction
│   │   ├── reasoner.go          # Autonomous chain-of-thought + tool loop
│   │   ├── planner.go           # Goal decomposition into step plans
│   │   ├── executor.go          # Tool execution + action recording
│   │   ├── reflector.go         # Quality monitoring + error detection
│   │   ├── learner.go           # Pattern extraction + persistence
│   │   ├── conversation.go      # Multi-turn history with auto-truncation
│   │   ├── persona.go           # System prompt + identity definition
│   │   ├── grounding.go         # Cognitive Grounding: budget, validation, reflection gate
│   │   ├── tool_selector.go     # Progressive tool disclosure by intent
│   │   ├── pipeline.go          # Cognitive Pipeline: fresh context per step
│   │   ├── router.go            # Multi-Model Router: task-based model selection
│   │   ├── session.go           # Session persistence (save / load / resume)
│   │   ├── scanner.go           # Project auto-detection (language, structure)
│   │   └── confirm.go           # User confirmation for dangerous actions
│   ├── memory/
│   │   ├── working.go           # Decay-based working memory (capacity-limited)
│   │   ├── longterm.go          # Persistent key-value long-term memory
│   │   ├── project.go           # Per-project fact store (conventions, decisions)
│   │   └── undo.go              # File modification undo stack
│   ├── ollama/
│   │   └── client.go            # Ollama HTTP client (chat, stream, ping, list)
│   ├── tools/
│   │   ├── registry.go          # Tool registry (register, get, list, describe)
│   │   └── builtin.go           # 18 built-in tools
│   ├── index/
│   │   └── codebase.go          # Go AST codebase index (functions, types, signatures)
│   └── compress/
│       └── atoms.go             # Context compression into reusable atoms
├── Makefile                     # build, run, test, lint, release
└── go.mod                       # Zero external dependencies
```

---

## Recommended Models

Nous works with any model available through Ollama. Smaller models are faster but less capable at complex reasoning and tool use. Larger models produce better results but need more RAM.

| Model | Parameters | RAM | Speed | Tool Use | Best For |
|---|---|---|---|---|---|
| `qwen2.5:1.5b` | 1.5B | ~2 GB | Fast | Decent | Quick tasks, light hardware (default) |
| `qwen2.5-coder:7b` | 7B | ~5 GB | Moderate | Good | Code generation and analysis |
| `codellama:7b` | 7B | ~5 GB | Moderate | Good | Code-focused tasks |
| `llama3.1:8b` | 8B | ~6 GB | Moderate | Good | General-purpose reasoning |
| `deepseek-coder-v2:16b` | 16B | ~12 GB | Slower | Strong | Complex multi-step work |
| `codestral:22b` | 22B | ~14 GB | Slower | Strong | Advanced code understanding |
| `qwen2.5:32b` | 32B | ~20 GB | Slow | Strong | When quality matters most |

Pull a model before first use:

```bash
ollama pull qwen2.5:1.5b
```

---

## Roadmap

- [x] Core cognitive architecture (6 concurrent streams as goroutines)
- [x] Blackboard pattern with event-driven pub/sub communication
- [x] Autonomous tool-use loop (up to 8 grounded calls per turn)
- [x] 18 built-in tools (read, write, edit, glob, grep, ls, tree, mkdir, shell, fetch, run, sysinfo, clipboard, find_replace, git, patch, replace_all, diff)
- [x] Streaming token output with tool-call JSON filtering
- [x] Session persistence and resume across restarts
- [x] Project auto-detection (language, file count, structure, key files)
- [x] Working memory with relevance decay and capacity eviction
- [x] Long-term memory with persistent JSON storage
- [x] Project memory — per-project persistent fact store
- [x] Undo stack — revert file modifications with `/undo`
- [x] Context compression via knowledge atoms
- [x] User confirmation for destructive actions (write, edit, shell, mkdir, patch, replace_all, find_replace)
- [x] Behavioral pattern learning from completed goals
- [x] Multi-platform release builds (Linux, macOS, amd64, arm64)
- [x] Git integration (status, diff, log, add, commit, branch)
- [x] HTTP fetch tool (local web scraping, no API)
- [x] System clipboard integration (read/write via xclip/xsel)
- [x] Regex find-and-replace with undo support
- [x] Multi-file string replacement across the codebase
- [x] Cognitive Grounding system (anti-hallucination for small models)
- [x] Progressive tool disclosure (intent-based tool selection)
- [x] Context budget tracking with auto-compression
- [x] Synchronous reflection gate (loop detection, forced convergence)
- [x] Cognitive Pipeline (fresh context per step, no degradation at step 8+)
- [x] Multi-Model Router (tinyllama for fast tasks, qwen for reasoning)
- [x] Codebase Index (Go AST parsing, 318 symbols, zero-cost structural context)
- [ ] Filesystem Sentinel (inotify-based ambient file watching)
- [ ] Tool Choreography (learned multi-step recipes)
- [ ] Predictive Pre-computation (speculative follow-up caching)
- [ ] Embedding-based atom retrieval (replace keyword overlap scoring)
- [ ] Automatic test generation and execution
- [ ] LSP integration for language-aware code navigation
- [ ] Voice input via local Whisper
- [ ] Plugin system for community tools
- [ ] TUI with split panes (conversation + file preview)

---

## Contributing

Contributions are welcome. The architecture is designed for extension without touching the core:

- **New tools** — Implement `func(args map[string]string) (string, error)` and register it in `internal/tools/builtin.go`. The reasoner discovers new tools automatically via the registry.
- **New cognitive streams** — Implement the `Stream` interface (`Name() string`, `Run(ctx context.Context) error`), subscribe to blackboard events, and add it to the stream list in `main.go`.
- **Bug fixes and improvements** — Standard PR workflow. Keep changes focused. Run `make test` and `make lint` before submitting.

The codebase has zero external Go dependencies. Let's keep it that way.

---

## License

MIT License — See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Nous</strong> — <em>I think, therefore I am — locally.</em>
</p>
<p align="center">
  Built by <a href="mailto:raphael.lugmayr@stoicera.com">Artaeon</a>
</p>
