package llm

import "context"

// Provider is the interface all LLM backends must implement.
type Provider interface {
	Chat(messages []Message, opts *Options) (*Response, error)
	ChatCtx(ctx context.Context, messages []Message, opts *Options) (*Response, error)
	Embed(text string) ([]float64, error)
	Model() string
}

// Message represents a single chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Options controls generation parameters.
type Options struct {
	Temperature float64
	TopP        float64
	MaxTokens   int
	Stop        []string
	NumCtx      int
}

// Response holds the result of a chat completion.
type Response struct {
	Content    string
	TokensUsed int
}
