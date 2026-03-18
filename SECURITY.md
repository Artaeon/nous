# Nous Security Model

## Threat Model

Nous runs locally and communicates only with a local Ollama server. The primary security concerns are:

1. **Tool execution safety** — preventing unintended file modifications or command execution
2. **Sandbox enforcement** — containing shell command effects
3. **Input validation** — handling malicious or malformed LLM outputs
4. **Network exposure** — securing the HTTP server mode

## Defense Layers

### 1. Dangerous Tool Classification

All tools are classified as safe or dangerous at registration time:

| Safe (no confirmation) | Dangerous (requires confirmation) |
|----------------------|----------------------------------|
| read, ls, tree, glob, grep | write, edit, patch, find_replace |
| sysinfo, diff, clipboard, fetch | shell, mkdir, run |
| git (read operations) | app (kill), process (kill) |
| weather, convert, currency, dict | volume, brightness |
| translate, hash, netcheck, qrcode | archive (extract/compress) |
| notes, todos, calendar, timer | notify, email |
| websearch, rss, summarize | screenshot, coderunner |
| diskusage, filefinder | |

When `--trust` flag is not set, dangerous tools require explicit user confirmation before execution. The confirmation prompt shows the action and details.

### 2. Execution Sandbox

The sandbox system (`internal/sandbox/`) provides:

- **Resource limits**: CPU time, memory, file descriptors
- **Policy enforcement**: Allow/deny lists for commands and paths
- **Output truncation**: Shell output capped at 8,192 bytes
- **Audit logging**: All sandbox decisions are logged

### 3. Predictor Safety

The predictive pre-computation system only executes **read-only tools**:
- `read`, `ls`, `tree`, `glob`, `grep`, `sysinfo`, `diff`
- Write tools (`write`, `edit`, `shell`, etc.) are never speculatively executed
- This is enforced by the `isReadOnly()` check (verified by tests)

### 4. Atomic File Operations

All persistent writes (recipes, memory, training data, sessions) use `safefile.WriteAtomic()`:
1. Write to a temporary file in the same directory
2. `fsync` the temporary file
3. Atomic `rename` over the target

This prevents data corruption from crashes or power loss.

### 5. HTTP Server Security

When running with `--serve`:
- **Default binding**: `127.0.0.1` (localhost only)
- **Public binding**: requires explicit `--public` flag to bind `0.0.0.0`
- **CORS**: validates Origin header, restricts to localhost
- **Timeouts**: Read 10s, Write 300s, Idle 60s
- **HTML escaping**: Web UI uses `escapeHtml()` guards on all dynamic content
- **Request serialization**: `chatMu` mutex prevents concurrent conversation state corruption

### 6. Reflection Gate (Anti-Looping)

The reflection gate prevents the LLM from entering infinite tool-call loops:
- Detects repeated tool calls via SHA256 hash comparison
- Escalates: 2 repeats → warning, 3 repeats → force stop
- Hard cap at 6 tool calls per reasoning cycle
- Consecutive empty results trigger early stopping (3 empty → force stop)

### 7. Context Budget Protection

The context budget system prevents token overflow attacks:
- Tracks estimated token usage per message
- Forces compression at 75% usage
- Forces immediate answer at 85% usage
- Prevents the LLM from consuming its own context window

## Privacy

- **No cloud**: All processing is local. No data leaves the machine.
- **No telemetry**: Nous collects no usage data.
- **No external APIs**: Only communicates with the local Ollama server.
- **Data location**: All persistent data stored in `~/.nous/` (configurable with `--memory`)

## Reporting Security Issues

If you discover a security vulnerability, please email raphael.lugmayr@stoicera.com directly rather than opening a public issue.
