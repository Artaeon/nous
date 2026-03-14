package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	DefaultOpenAIURL = "https://api.openai.com/v1"
	defaultTimeout   = 300 * time.Second
)

// OpenAIProvider implements Provider for any OpenAI-compatible API.
// Works with OpenAI, Groq, Together, OpenRouter, vLLM, LM Studio, etc.
type OpenAIProvider struct {
	baseURL    string
	apiKey     string
	model      string
	httpClient *http.Client
}

// NewOpenAI creates a provider for an OpenAI-compatible API endpoint.
func NewOpenAI(baseURL, apiKey, model string) *OpenAIProvider {
	if baseURL == "" {
		baseURL = DefaultOpenAIURL
	}
	return &OpenAIProvider{
		baseURL: baseURL,
		apiKey:  apiKey,
		model:   model,
		httpClient: &http.Client{
			Timeout: defaultTimeout,
		},
	}
}

// Model returns the configured model name.
func (p *OpenAIProvider) Model() string {
	return p.model
}

// --- OpenAI API types ---

type openaiRequest struct {
	Model       string          `json:"model"`
	Messages    []openaiMessage `json:"messages"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	MaxTokens   *int            `json:"max_tokens,omitempty"`
	Stop        []string        `json:"stop,omitempty"`
	Stream      bool            `json:"stream"`
}

type openaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openaiResponse struct {
	ID      string         `json:"id"`
	Choices []openaiChoice `json:"choices"`
	Usage   openaiUsage    `json:"usage"`
	Error   *openaiError   `json:"error,omitempty"`
}

type openaiChoice struct {
	Index        int           `json:"index"`
	Message      openaiMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

type openaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openaiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

type openaiEmbedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type openaiEmbedResponse struct {
	Data  []openaiEmbedData `json:"data"`
	Error *openaiError      `json:"error,omitempty"`
}

type openaiEmbedData struct {
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// Chat sends a chat completion request and returns the response.
func (p *OpenAIProvider) Chat(messages []Message, opts *Options) (*Response, error) {
	return p.ChatCtx(context.Background(), messages, opts)
}

// ChatCtx sends a chat completion request with context support.
func (p *OpenAIProvider) ChatCtx(ctx context.Context, messages []Message, opts *Options) (*Response, error) {
	oaiMessages := make([]openaiMessage, len(messages))
	for i, m := range messages {
		oaiMessages[i] = openaiMessage{Role: m.Role, Content: m.Content}
	}

	req := openaiRequest{
		Model:    p.model,
		Messages: oaiMessages,
		Stream:   false,
	}

	if opts != nil {
		if opts.Temperature > 0 {
			t := opts.Temperature
			req.Temperature = &t
		}
		if opts.TopP > 0 {
			t := opts.TopP
			req.TopP = &t
		}
		if opts.MaxTokens > 0 {
			m := opts.MaxTokens
			req.MaxTokens = &m
		}
		if len(opts.Stop) > 0 {
			req.Stop = opts.Stop
		}
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("openai request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result openaiResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	if result.Error != nil {
		return nil, fmt.Errorf("openai error: %s (type: %s)", result.Error.Message, result.Error.Type)
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("openai returned no choices")
	}

	return &Response{
		Content:    result.Choices[0].Message.Content,
		TokensUsed: result.Usage.TotalTokens,
	}, nil
}

// Embed generates an embedding vector for the given text.
func (p *OpenAIProvider) Embed(text string) ([]float64, error) {
	req := openaiEmbedRequest{
		Model: p.model,
		Input: text,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(context.Background(), "POST", p.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create embed request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("embed request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read embed response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result openaiEmbedResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}

	if result.Error != nil {
		return nil, fmt.Errorf("embed error: %s", result.Error.Message)
	}

	if len(result.Data) == 0 {
		return nil, fmt.Errorf("embed returned no data")
	}

	return result.Data[0].Embedding, nil
}
