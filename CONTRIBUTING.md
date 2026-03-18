# Contributing to Nous

Thank you for your interest in contributing to Nous! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Go 1.22 or later
- [Ollama](https://ollama.ai) installed and running
- A local model pulled (e.g., `ollama pull qwen2.5:1.5b`)

### Setup

```bash
git clone https://github.com/artaeon/nous.git
cd nous
go build -o nous ./cmd/nous
go test ./...
```

### Running Tests

```bash
# All tests
go test ./...

# Verbose with count
go test ./... -v -count=1

# Specific package
go test ./internal/cognitive/ -v

# With race detection
go test ./... -race

# With coverage
go test ./... -coverprofile=cover.out
go tool cover -html=cover.out
```

## Project Principles

### Zero External Dependencies

Nous uses **only the Go standard library**. This is a hard constraint. Do not add external dependencies. If you need functionality from an external package, implement it yourself using the stdlib.

Why: A single static binary with no dependency tree means no supply chain attacks, no version conflicts, no `go mod tidy` surprises, and trivial deployment.

### Architecture

Nous follows a **cognitive architecture** pattern:

- **Streams** are independent goroutines that process events
- **Blackboard** is the shared workspace for inter-stream communication
- **Tools** are the capabilities the reasoner can invoke
- **Memory** layers provide short and long-term storage

When adding features, consider which layer they belong to.

### Code Style

- Standard `go fmt` formatting
- No comments on obvious code
- Comments for non-obvious logic, public APIs, and architectural decisions
- Test files alongside source files (`foo.go` + `foo_test.go`)
- Package names should be short, lowercase, single-word

## How to Contribute

### Bug Reports

Open an issue with:
1. What you expected to happen
2. What actually happened
3. Steps to reproduce
4. Your environment (OS, Go version, Ollama version, model)

### Feature Requests

Open an issue describing:
1. The problem you're trying to solve
2. Your proposed solution
3. Alternatives you considered

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run `go test ./...` and `go vet ./...`
6. Commit with a descriptive message
7. Push and open a PR

#### PR Guidelines

- Keep PRs focused on a single change
- Add tests for new code
- Update the README if adding user-facing features
- Don't break existing tests
- Prefer small, incremental changes over large rewrites

## Areas for Contribution

### High Impact

- **Language support for codebase index** &mdash; Currently Go-only. Adding Python, TypeScript, Rust AST parsing would be very valuable.
- **Streaming in server mode** &mdash; Server-Sent Events for real-time token streaming
- **Better prediction strategies** &mdash; More intelligent speculative pre-computation
- **Model-specific optimizations** &mdash; Tuning prompts and parameters for specific models (llama3, mistral, phi, etc.)

### Medium Impact

- **More tools** &mdash; Docker, Kubernetes, database clients, HTTP testing, Bluetooth, Wi-Fi management
- **macOS/Windows sentinel** &mdash; FSEvents and ReadDirectoryChanges equivalents
- **Fine-tuning datasets** &mdash; Curated training data for common coding tasks
- **Web UI improvements** &mdash; Markdown rendering, code highlighting, file browser

### Good First Issues

- Add a new slash command
- Improve error messages
- Add test cases for edge cases
- Documentation improvements

## Release Process

Versions follow semantic versioning: `MAJOR.MINOR.PATCH`

- `MAJOR` &mdash; Breaking changes to CLI flags or API
- `MINOR` &mdash; New features, new tools, new systems
- `PATCH` &mdash; Bug fixes, performance improvements

## Code of Conduct

Be respectful, be constructive, be kind. We're building something together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
