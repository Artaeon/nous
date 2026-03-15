# Nous Architecture

A deep dive into the cognitive architecture that makes Nous work with small local models.

## The Core Problem

Large language models (GPT-4, Claude) can reason over long contexts and self-correct. Small models (1.5B-7B parameters) cannot. They hallucinate, lose track of context, repeat themselves, and fail at multi-step reasoning. Nous solves this architecturally rather than by throwing more parameters at the problem.

## Design Philosophy

**1. Fresh context beats accumulated context.**
Small models degrade severely when >70% of their context window is consumed. Instead of accumulating messages like a chatbot, Nous gives each reasoning step a fresh context with compressed summaries of prior steps. This is the single most important architectural decision.

**2. Architecture compensates for model limitations.**
Where large models can self-correct, Nous uses explicit systems: a reflection gate catches repetition, a grounding system validates tool results, and a context budget prevents overflow.

**3. Zero dependencies, zero cloud.**
The entire system is pure Go stdlib. No external packages, no API keys, no network calls beyond the local Ollama server. This is an ideological choice: Nous runs entirely on your hardware.

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                     User Input                          │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Query Classifier│ ← 3-tier: fast/medium/full
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐  ┌──────▼──────┐  ┌───▼───────────┐
   │Fast Path│  │Medium Path  │  │Full Pipeline  │
   │(1 call) │  │(1 call+mem) │  │(6 streams)    │
   └─────────┘  └─────────────┘  └───┬───────────┘
                                     │
                       ┌─────────────┼─────────────┐
                       │             │             │
                  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
                  │Perceiver│  │Reasoner │  │Planner  │
                  │(intent) │  │(tools)  │  │(goals)  │
                  └────┬────┘  └────┬────┘  └────┬────┘
                       │            │            │
                  ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
                  │Executor │  │Reflector│  │Learner  │
                  │(actions)│  │(quality)│  │(patterns│
                  └─────────┘  └─────────┘  └─────────┘
                       │
              ┌────────▼────────┐
              │   Blackboard    │ ← Shared pub/sub workspace
              └─────────────────┘
```

## Key Innovations

### 1. Fresh-Context Pipeline

**Problem**: At step 8+, small models have consumed so much context that they start hallucinating or repeating previous tool calls.

**Solution**: Each reasoning step gets a fresh LLM call containing only:
1. A compact system prompt with tool definitions
2. The original user question
3. One-line compressed summaries of all previous steps
4. The current tool result

Previous step results are compressed using either:
- **Thought distillation**: A fast model (tinyllama) summarizes the result in one sentence
- **Rule-based compression**: Deterministic heuristics per tool type (e.g., read→"Read file.go: first line... (N lines)")

This keeps context usage under 30% even at step 10+, compared to 90%+ for accumulated-message approaches.

**Benchmark**: Pipeline context build takes 1.9μs — negligible vs LLM inference.

### 2. 5-Layer Cognitive Grounding

Each layer prevents a different class of hallucination:

| Layer | What it prevents | How |
|-------|-----------------|-----|
| **Context Budget** | Token overflow | Tracks chars-per-token estimation; triggers compression at 75%, forced answer at 85% |
| **Smart Truncation** | Information overload | Tool-specific truncation preserving navigability (landmarks from middle of files, line numbers) |
| **Result Validation** | Acting on errors | Checks for empty/error results, provides corrective hints ("Path not found. Use ls to find the correct path.") |
| **Reflection Gate** | Repetition loops | SHA256 hash of results in 4-entry circular buffer; detects repeated tool calls and escalates: nudge→warn→force stop |
| **Iteration Cap** | Runaway reasoning | Hard stop at 6 iterations with forced answer from accumulated evidence |

**Benchmark**: Reflection gate check takes 418ns — zero perceptible overhead.

### 3. 3-Tier Query Classification

Not every query needs the full cognitive pipeline. Nous classifies queries in <3μs using compiled regex patterns:

| Tier | Latency | Context | Examples |
|------|---------|---------|----------|
| **Fast** | ~50ms | System prompt + query | "hi", "thanks", "who are you?", "42+7?" |
| **Medium** | ~200ms | + conversation history + memory | "explain how GC works", "tell me more", "what do you think?" |
| **Full** | ~2-10s | Full 6-stream pipeline with tools | "read file main.go", "find all TODOs", "refactor the server" |

This saves 2-10 seconds on simple queries while still handling complex tasks.

### 4. Predictive Pre-computation

After each tool execution, Nous speculatively caches likely follow-up results:

| Last Action | Prediction | Hit Rate |
|------------|-----------|----------|
| Read `X.go` | Pre-read `X_test.go` | ~40% |
| Read file | Pre-list its directory | ~25% |
| Grep for symbol | Pre-read first matched file | ~60% |
| List directory | Pre-read README.md, main.go | ~30% |
| Glob for pattern | Pre-read first 2 matched files | ~35% |

Predictions run in background goroutines, are read-only (safe), expire after 30 seconds, and the cache is capped at 20 entries with LRU eviction.

**Benchmark**: Cache lookup takes 519ns. On a hit, this saves 50-500ms of tool execution.

### 5. Tool Choreography (Recipes)

Nous learns successful multi-step tool sequences and replays them:

1. **Record**: After a successful 2+ step pipeline, extract the tool sequence
2. **Parameterize**: Replace concrete paths with `$FILE`, `$DIR` placeholders
3. **Match**: On new queries, score recipes by intent + keyword overlap + confidence
4. **Replay**: Substitute parameters and execute the learned sequence

Example learned recipe:
```
Trigger: "find function definition"
Sequence: grep($PATTERN) → read($FILE)
Confidence: 0.87 (successes/uses)
```

Recipes are persisted to disk and pruned to keep the top 40 by confidence.

### 6. Multi-Model Routing

Different cognitive tasks have different requirements:

| Task | Model | Why |
|------|-------|-----|
| Perception | tinyllama | Fast, simple intent extraction |
| Compression | tinyllama | Quick summarization |
| Reasoning | qwen2.5:1.5b | Best accuracy for tool selection |
| Reflection | qwen2.5:1.5b | Needs judgment |

The router auto-discovers available models via the Ollama API, classifies them by family (size, capabilities), and routes tasks accordingly. Falls back to the default model if specialized models aren't available.

### 7. Self-Improvement Pipeline

Nous collects every successful interaction as training data:

```
Interaction → Quality Score (0-1) → Filter (≥0.6) → Collect → Export
                                                         ↓
                                              JSONL / Alpaca / ChatML
                                                         ↓
                                              LoRA fine-tuning (unsloth)
                                                         ↓
                                              Enhanced Modelfile
                                                         ↓
                                              ollama create nous-tuned
```

Auto-tuning triggers when: 50+ pairs collected, average quality ≥0.7, and 1-hour cooldown since last tune. The tuned model embeds learned patterns into its system prompt.

## Memory Architecture

```
┌──────────────────────────────────────────┐
│              Working Memory               │
│  64 slots, decay-based (τ=0.95/step)     │
│  Semantic embeddings for relevance       │
└──────────────┬───────────────────────────┘
               │ promotes
┌──────────────▼───────────────────────────┐
│            Long-Term Memory               │
│  Persistent JSON key-value store         │
│  Categorized entries (facts, prefs)      │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│            Project Memory                 │
│  Per-project facts in .nous/ directory   │
│  Language, framework, conventions        │
└──────────────────────────────────────────┘
┌──────────────────────────────────────────┐
│           Episodic Memory                 │
│  Every interaction with embeddings       │
│  Hybrid search: semantic + keyword       │
│  Auto-prune at 10K episodes              │
└──────────────────────────────────────────┘
```

## Concurrency Model

All cognitive streams run as independent goroutines communicating through the blackboard (pub/sub). The blackboard uses `sync.RWMutex` for thread-safe access. Each stream:

1. Subscribes to specific blackboard topics
2. Processes events independently
3. Publishes results back to the blackboard
4. Can be cancelled via context

The predictor cache, recipe book, episodic memory, and training collector all use `RWMutex` for concurrent read/write safety. All pass Go's race detector (`-race` flag).

## File Organization

```
cmd/nous/main.go          → Entry point, REPL, server, slash commands (2,756 lines)
internal/
  cognitive/
    reasoner.go            → Core agent: tool calling, reasoning loop (1,886 lines)
    pipeline.go            → Fresh-context step management (350 lines)
    grounding.go           → 5-layer anti-hallucination (242 lines)
    fastpath.go            → 3-tier query classification (275 lines)
    predictor.go           → Speculative pre-computation (315 lines)
    recipes.go             → Tool choreography learning (428 lines)
    router.go              → Multi-model task routing (336 lines)
    diffpreview.go         → LCS-based colored diffs (298 lines)
    confirm.go             → Tool safety classification (53 lines)
    persona.go             → Identity and prompts (66 lines)
    + 15 more files
  ollama/client.go         → Ollama HTTP client with native tool calling (498 lines)
  memory/                  → 4-layer memory system (1,468 lines)
  tools/                   → 18 built-in tools + browser (2,618 lines)
  server/                  → HTTP API + web UI (931 lines)
  training/                → LoRA fine-tuning pipeline (719 lines)
  sentinel/                → inotify filesystem watcher (342 lines)
  index/                   → Go AST codebase indexer (681 lines)
  + 8 more packages
```

Total: **23,843 lines** of production Go code, **19,643 lines** of tests, **zero external dependencies**.
