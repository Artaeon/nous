# Changelog

All notable changes to Nous are documented here.

## [1.0.0] - 2026-03-21

### Revolutionary
- **Pure Cognitive Engine** — Nous now runs entirely without any external model dependency
- **Thinking Engine** — 12 task types (compose, brainstorm, analyze, teach, advise, compare, summarize, create, plan, debate, reflect, converse) with frame-based generation
- **Discourse Planner** — RST-based rhetorical structure planning for coherent text generation
- **Frame System** — 12 structural templates for different output types
- **Built-in Embeddings** — 50-dimensional word vectors replace external embedding models
- **Compositional Generation** — 10 clause patterns, 6 realization strategies, Markov chains
- **Knowledge Packages** — JSON-based instant domain knowledge loading

### Removed
- Ollama dependency — no longer required at runtime
- External model downloads — no qwen, nomic-embed-text, or tinyllama needed
- GPU requirement — runs on any hardware with ~50 MB RAM

### Added
- 6 new tools: calculator, password generator, bookmarks, journal, habits, expenses (51 total)
- Multi-intent NLU parsing for compound queries
- Fuzzy NLU matching with Levenshtein distance and synonym expansion
- Creative writing: poetry (haiku + free verse), stories, concept blending, analogy generation
- Tone-aware generation (formal, casual, neutral, academic, empathetic)

## [0.9.0] - 2026-03-18

### Added
- **Deterministic NLU Engine**: Rule-based intent classification with 30+ categories, entity extraction, and confidence scoring — routes most queries in <1ms with pure deterministic processing
- **ActionRouter**: Direct tool dispatch from NLU results, returning DirectResponse for deterministic query handling
- **27 new assistant tools** (total: 45 built-in tools):
  - **Batch 1**: weather, convert, currency, notes, todos, filefinder, summarize, rss, coderunner, calendar, email, screenshot, websearch
  - **Batch 2**: volume, brightness, notify, timer, app launcher, hash/encode, dictionary, netcheck, translate, qrcode, archive, diskusage, process manager
- **handleGenericTool**: Reusable action handler pattern for simple tool delegation
- **WeaveForPath**: Path-aware context assembly that filters code-indexed content from Q&A queries
- **Response crystal normalization**: Strips question preambles for better cache hit rates
- **Reverse conversion parsing**: Regex support for "how many km is 10 miles" format
- **NLU word lists**: 20+ domain-specific word lists for deterministic intent matching (weatherWords, convertWords, volumeWords, brightnessWords, timerWords, translateWords, etc.)

### Changed
- Tools expanded from 18 to 45
- Binary size: ~14 MB (from ~11 MB)
- NLU checked before FastPath in REPL — most queries now handled deterministically
- Knowledge source relevance threshold raised from 0.3 to 0.5 (reduces noise)
- Quick greetings expanded from 23 to 68 entries
- Deterministic processing rate on day 1: ~80% (up from ~65%)
- Web search returns top 3 formatted results directly
- Memory lookups return single facts directly
- Schedule/reminder actions return formatted confirmations directly

### Fixed
- NLU misrouting: assistant features now checked before generic verb rules
- Timer/reminder word conflict: "set a timer" no longer routes to reminders
- Knowledge contamination: code-indexed content filtered from Q&A responses
- CLI not routing through NLU: all channels now share NLU/ActionRouter
- File finder case sensitivity on Linux for explicit paths
- External IP response leaking HTML on some networks
- Schedule action returning empty task text

## [0.6.0] - 2026-03-11

### Added
- **Episodic Memory**: Stores every interaction forever with embedding-based semantic search (cosine similarity over built-in word vectors)
- **Training Data Pipeline**: Collects successful interactions as JSONL/Alpaca/ChatML (optional, for advanced use)
- **Fine-tune Script Generator**: Complete Python script for optional QLoRA training with unsloth (advanced)
- **HTTP Server Mode**: `--serve` flag runs Nous as an HTTP API with web UI
- **Filesystem Sentinel**: inotify-based ambient file watching, auto-updates codebase index on changes
- **Tool Choreography**: Records successful multi-step tool sequences as reusable recipes
- **Predictive Pre-computation**: Speculatively pre-executes likely follow-up tool calls
- **Docker Support**: Dockerfile + docker-compose for containerized deployment
- **Systemd Service**: Production-ready unit file with security hardening
- **Install Script**: One-line installer for Linux servers
- **Built-in Embeddings**: 50-dimensional word vectors for semantic search
- New commands: `/episodes`, `/search`, `/training`, `/export`, `/finetune`

### Changed
- Version bumped to 0.6.0
- Binary size: 9.9 MB (from 9.3 MB)
- Test count: 316 (from 283)
- Persona updated with unrestricted assistant identity
- Makefile expanded with `serve`, `install`, `docker`, `release` targets

## [0.5.0] - 2026-03-10

### Added
- **Cognitive Pipeline**: Fresh context per reasoning step eliminates degradation after 3-4 tool calls
- **Cognitive Router**: Routes queries through appropriate cognitive processing paths
- **Codebase Index**: Go AST parsing extracts every function, struct, method with signatures
- **Cognitive Grounding**: 5-layer anti-hallucination (progressive disclosure, smart truncation, validation, budget tracking, reflection gate)
- **Progressive Tool Disclosure**: Shows 5-8 relevant tools per intent instead of all 18
- **Context Budget**: Token tracking with auto-compression at 75% and forced answer at 85%
- **Reflection Gate**: Detects loops, repetition, consecutive failures; forces convergence

### Changed
- Reasoner rewritten with pipeline architecture
- System prompt ordering: tool instructions first, identity last
- Max tool iterations reduced from 15 to 8

## [0.4.0] - 2026-03-09

### Added
- 18 tools: added `patch`, `find_replace`, `replace_all`, `diff`, `git`, `clipboard`, `fetch`, `run`, `sysinfo`
- File modification undo stack
- Project memory (`.nous/` per-project fact store)
- Session persistence and resume

## [0.3.0] - 2026-03-08

### Added
- 9 built-in tools: `read`, `write`, `edit`, `glob`, `grep`, `ls`, `tree`, `mkdir`, `shell`
- Working memory (decay-based, 64 slots)
- Long-term memory (persistent JSON)
- Confirmation prompts for dangerous actions
- Trust mode (`--trust`)

## [0.2.0] - 2026-03-07

### Added
- Six cognitive streams (perceive, reason, plan, execute, reflect, learn)
- Blackboard architecture with event-driven pub/sub
- Cognitive engine with streaming output
- Basic REPL with slash commands

## [0.1.0] - 2026-03-06

### Added
- Initial project structure
- Core cognitive engine with chat and streaming
- Basic reasoning loop
