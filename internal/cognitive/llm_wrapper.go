package cognitive

import (
	"context"
	"strings"
	"time"

	"github.com/artaeon/nous/internal/ollama"
)

// OllamaLLM wraps the Ollama client to implement LLMClient.
type OllamaLLM struct {
	Client *ollama.Client
}

// Generate sends a prompt to Ollama and returns the response.
// Uses optimized parameters: small context window (512) and tight token
// limit for fast responses on small models.
func (o *OllamaLLM) Generate(system, prompt string, maxTokens int) string {
	if o == nil || o.Client == nil {
		return ""
	}
	if maxTokens <= 0 {
		maxTokens = 128
	}

	// Timeout scales with token count: ~100ms per token on 1.5B model
	timeout := time.Duration(maxTokens/2+5) * time.Second
	if timeout < 10*time.Second {
		timeout = 10 * time.Second
	}
	if timeout > 30*time.Second {
		timeout = 30 * time.Second
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	var messages []ollama.Message
	if system != "" {
		messages = append(messages, ollama.Message{Role: "system", Content: system})
	}
	messages = append(messages, ollama.Message{Role: "user", Content: prompt})

	resp, err := o.Client.ChatCtx(ctx, messages, &ollama.ModelOptions{
		Temperature: 0.3,
		NumPredict:  maxTokens,
	})
	if err != nil {
		return ""
	}

	return strings.TrimSpace(resp.Message.Content)
}

// GenerateStream sends a prompt and streams tokens via callback.
// Returns the full response text.
func (o *OllamaLLM) GenerateStream(system, prompt string, maxTokens int, onToken func(string)) string {
	if o == nil || o.Client == nil {
		return ""
	}
	if maxTokens <= 0 {
		maxTokens = 128
	}

	var messages []ollama.Message
	if system != "" {
		messages = append(messages, ollama.Message{Role: "system", Content: system})
	}
	messages = append(messages, ollama.Message{Role: "user", Content: prompt})

	resp, err := o.Client.ChatStream(messages, &ollama.ModelOptions{
		Temperature: 0.3,
		NumPredict:  maxTokens,
	}, func(token string, done bool) {
		if onToken != nil && token != "" {
			onToken(token)
		}
	})
	if err != nil {
		return ""
	}

	return strings.TrimSpace(resp.Message.Content)
}

// ModelName returns the Ollama model name.
func (o *OllamaLLM) ModelName() string {
	if o == nil || o.Client == nil {
		return ""
	}
	return o.Client.Model()
}
