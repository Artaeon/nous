# Changelog

All notable changes to Nous are documented here.

## [0.6.0] - 2026-03-11

### Added
- **Episodic Memory**: Stores every interaction forever with embedding-based semantic search (cosine similarity over Ollama vectors)
- **Training Data Pipeline**: Collects successful interactions as JSONL/Alpaca/ChatML for LoRA fine-tuning
- **Modelfile Generator**: Creates custom Ollama models with personality baked into the weights
- **Fine-tune Script Generator**: Complete Python script for QLoRA training with unsloth
- **HTTP Server Mode**: `--serve` flag runs Nous as an HTTP API with web UI
- **Filesystem Sentinel**: inotify-based ambient file watching, auto-updates codebase index on changes
- **Tool Choreography**: Records successful multi-step tool sequences as reusable recipes
- **Predictive Pre-computation**: Speculatively pre-executes likely follow-up tool calls
- **Docker Support**: Dockerfile + docker-compose with Ollama and GPU passthrough
- **Systemd Service**: Production-ready unit file with security hardening
- **Install Script**: One-line installer for Linux servers
- **Ollama Embeddings API**: Client support for `/api/embeddings` endpoint
- **Ollama Create Model API**: Client support for `/api/create` endpoint
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
- **Multi-Model Router**: Auto-discovers Ollama models, routes perception to tinyllama, reasoning to qwen
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
- Ollama HTTP client with streaming
- Basic REPL with slash commands

## [0.1.0] - 2026-03-06

### Added
- Initial project structure
- Ollama client with chat and streaming
- Basic reasoning loop
