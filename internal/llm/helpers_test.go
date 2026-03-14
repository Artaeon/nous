package llm

import "github.com/artaeon/nous/internal/ollama"

// newTestOllamaClient creates an ollama.Client pointing at a test server.
func newTestOllamaClient(host, model string) *ollama.Client {
	return ollama.New(
		ollama.WithHost(host),
		ollama.WithModel(model),
	)
}
