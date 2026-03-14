package llm

import (
	"context"

	"github.com/artaeon/nous/internal/ollama"
)

// OllamaProvider wraps an existing ollama.Client to implement the Provider interface.
type OllamaProvider struct {
	client *ollama.Client
}

// NewOllama creates a Provider that delegates to an existing ollama.Client.
func NewOllama(client *ollama.Client) *OllamaProvider {
	return &OllamaProvider{client: client}
}

// Model returns the configured model name.
func (o *OllamaProvider) Model() string {
	return o.client.Model()
}

// Chat sends messages to Ollama and returns the response.
func (o *OllamaProvider) Chat(messages []Message, opts *Options) (*Response, error) {
	ollamaMessages := convertMessages(messages)
	ollamaOpts := convertOptions(opts)

	resp, err := o.client.Chat(ollamaMessages, ollamaOpts)
	if err != nil {
		return nil, err
	}

	return &Response{
		Content:    resp.Message.Content,
		TokensUsed: resp.PromptEvalCount + resp.EvalCount,
	}, nil
}

// ChatCtx sends messages to Ollama with context support.
func (o *OllamaProvider) ChatCtx(ctx context.Context, messages []Message, opts *Options) (*Response, error) {
	ollamaMessages := convertMessages(messages)
	ollamaOpts := convertOptions(opts)

	resp, err := o.client.ChatCtx(ctx, ollamaMessages, ollamaOpts)
	if err != nil {
		return nil, err
	}

	return &Response{
		Content:    resp.Message.Content,
		TokensUsed: resp.PromptEvalCount + resp.EvalCount,
	}, nil
}

// Embed generates an embedding vector for the given text.
func (o *OllamaProvider) Embed(text string) ([]float64, error) {
	return o.client.Embed(text)
}

// Client returns the underlying ollama.Client for direct access.
func (o *OllamaProvider) Client() *ollama.Client {
	return o.client
}

func convertMessages(messages []Message) []ollama.Message {
	out := make([]ollama.Message, len(messages))
	for i, m := range messages {
		out[i] = ollama.Message{Role: m.Role, Content: m.Content}
	}
	return out
}

func convertOptions(opts *Options) *ollama.ModelOptions {
	if opts == nil {
		return nil
	}
	o := &ollama.ModelOptions{}
	if opts.Temperature > 0 {
		o.Temperature = opts.Temperature
	}
	if opts.TopP > 0 {
		o.TopP = opts.TopP
	}
	if opts.MaxTokens > 0 {
		o.NumPredict = opts.MaxTokens
	}
	if opts.NumCtx > 0 {
		o.NumCtx = opts.NumCtx
	}
	if len(opts.Stop) > 0 {
		o.Stop = opts.Stop
	}
	return o
}
