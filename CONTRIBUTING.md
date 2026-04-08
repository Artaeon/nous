# Contributing to NOUS

Thank you for your interest in contributing to NOUS (Native Orchestration of Unified Streams). This document provides guidelines for contributors.

## Getting Started

### Prerequisites

- Go 1.22 or later
- That's it — no external services, model downloads, or system libraries

### Setup

```bash
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous
go test -short ./...
go vet ./...
```

### Running Tests

```bash
# All tests (skip slow neural evaluation)
go test -short ./...

# Verbose with count
go test ./... -v -count=1

# Specific package
go test ./internal/cognitive/ -v -run "TestSimulation"

# With race detection
go test ./... -race

# With coverage
go test ./... -coverprofile=cover.out
go tool cover -html=cover.out
```

## Project Principles

### Zero External Dependencies

NOUS uses **only the Go standard library**. This is a hard constraint. Do not add external dependencies. If you need functionality from an external package, implement it using the stdlib.

Why: A single static binary with no dependency tree means no supply chain attacks, no version conflicts, no `go mod tidy` surprises, and trivial deployment (`scp` + `./nous`).

### Cognitive Architecture

NOUS follows a **6-stream cognitive architecture**:

- **Perceiver** — input parsing, entity extraction, intent classification
- **Reasoner** — knowledge graph traversal, causal inference, multi-hop reasoning
- **Planner** — goal decomposition, phase planning, tool chain selection
- **Executor** — tool dispatch, result collection, error recovery
- **Reflector** — quality evaluation, pattern detection, self-assessment
- **Learner** — cognitive compilation, conversation learning, knowledge expansion

All six streams run concurrently on a shared **blackboard** (pub/sub event bus). When adding features, consider which stream and which cognitive layer they belong to.

### Key Subsystems

| Subsystem | Package | What it does |
|-----------|---------|-------------|
| **Cognitive Engine** | `internal/cognitive/` | 175 modules — NLU, NLG, knowledge graph, reasoning, compiler |
| **Agent Framework** | `internal/agent/` | Goal decomposition, tool chaining, experience learning |
| **Mamba SSM** | `internal/micromodel/` | Pure-Go neural model with constrained decoding |
| **Memory** | `internal/memory/` | 6-layer persistent memory system |
| **Tools** | `internal/tools/` | 52 built-in tools with undo support |
| **Federation** | `internal/federation/` | Privacy-preserving crystal sharing |

### Code Style

- Standard `go fmt` formatting
- No comments on obvious code
- Comments for non-obvious logic, public APIs, and architectural decisions
- Test files alongside source files (`foo.go` + `foo_test.go`)
- Package names: short, lowercase, single-word
- CI pipeline: `go vet ./...` must pass with zero warnings

## How to Contribute

### Bug Reports

Open an issue with:
1. What you expected to happen
2. What actually happened
3. Steps to reproduce
4. Your environment (OS, Go version, `nous --version` output)

### Feature Requests

Open an issue describing:
1. The problem you're trying to solve
2. Your proposed solution
3. Alternatives you considered
4. Which cognitive layer it affects

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run `go test -short ./...` and `go vet ./...`
6. Commit with a descriptive message
7. Push and open a PR

#### PR Guidelines

- Keep PRs focused on a single change
- Add tests for new code
- Update README.md if adding user-facing features
- Don't break existing tests
- `go vet ./...` must pass cleanly
- Prefer small, incremental changes over large rewrites

## Areas for Contribution

### High Impact — Cognitive Systems

- **Simulation depth** — Richer multi-step causal chains, scenario branching, Monte Carlo sampling
- **Dream Mode quality** — Better novelty detection, convergence checks, insight synthesis
- **Knowledge Synthesis** — Stronger analogical reasoning, compositional inference
- **Mamba training** — Causal chain training data, longer context, better generation quality
- **Prose fluency** — Template variety, sentence fusion, discourse planning improvements

### High Impact — Infrastructure

- **Language support for codebase index** — Currently Go-only. Python, TypeScript, Rust AST parsing
- **Streaming in server mode** — Server-Sent Events for real-time token streaming
- **Multi-turn conversation** — Better reference resolution, context carryover

### Medium Impact

- **More tools** — Docker, Kubernetes, database clients, HTTP testing
- **Knowledge packages** — New domain packages for specialized topics (medicine, law, finance)
- **Expert personas** — New domain experts, deeper domain constraints
- **macOS/Windows sentinel** — FSEvents and ReadDirectoryChanges equivalents
- **Web UI** — Markdown rendering, code highlighting, knowledge graph visualization

### Good First Issues

- Add a new slash command
- Add test cases for edge cases in NLU classification
- Improve error messages in tool execution
- Expand the causal bootstrap knowledge (add edges for new domains)
- Add more emotional response variety

## Release Process

Versions follow semantic versioning: `MAJOR.MINOR.PATCH`

- `MAJOR` — Breaking changes to CLI flags, API, or data formats
- `MINOR` — New cognitive systems, new tools, new capabilities
- `PATCH` — Bug fixes, performance improvements, quality improvements

## Code of Conduct

Be respectful, be constructive, be kind. We're building something together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
