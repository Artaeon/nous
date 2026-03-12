package ollama

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	DefaultHost    = "http://localhost:11434"
	DefaultModel   = "qwen2.5:1.5b"
	DefaultTimeout = 120 * time.Second
)

type Client struct {
	host       string
	model      string
	httpClient *http.Client
}

type Option func(*Client)

func WithHost(host string) Option {
	return func(c *Client) { c.host = host }
}

func WithModel(model string) Option {
	return func(c *Client) { c.model = model }
}

func WithTimeout(timeout time.Duration) Option {
	return func(c *Client) { c.httpClient.Timeout = timeout }
}

func New(opts ...Option) *Client {
	c := &Client{
		host:  DefaultHost,
		model: DefaultModel,
		httpClient: &http.Client{
			Timeout: DefaultTimeout,
		},
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// --- Message Types ---

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"` // present when assistant calls tools
	ToolName  string     `json:"tool_name,omitempty"`  // set when role=="tool"
}

// ToolCall represents a tool invocation returned by the model.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction contains the function name and parsed arguments.
type ToolCallFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// --- Tool Definition Types (for requests) ---

// Tool represents a tool the model can call via the native API.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a callable function.
type ToolFunction struct {
	Name        string               `json:"name"`
	Description string               `json:"description"`
	Parameters  ToolFunctionParams   `json:"parameters"`
}

// ToolFunctionParams is the JSON Schema for function parameters.
type ToolFunctionParams struct {
	Type       string                  `json:"type"`
	Properties map[string]ToolProperty `json:"properties"`
	Required   []string                `json:"required,omitempty"`
}

// ToolProperty describes a single parameter in the JSON Schema.
type ToolProperty struct {
	Type        string   `json:"type"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
}

// ToolResultMessage creates a message to send tool results back to the model.
func ToolResultMessage(toolName, content string) Message {
	return Message{
		Role:     "tool",
		Content:  content,
		ToolName: toolName,
	}
}

// --- Request/Response Types ---

type GenerateRequest struct {
	Model    string        `json:"model"`
	Messages []Message     `json:"messages"`
	Stream   bool          `json:"stream"`
	Options  *ModelOptions `json:"options,omitempty"`
	Tools    []Tool        `json:"tools,omitempty"`
	Format   string        `json:"format,omitempty"` // "json" constrains output to valid JSON
}

type ModelOptions struct {
	Temperature   float64  `json:"temperature,omitempty"`
	TopP          float64  `json:"top_p,omitempty"`
	NumCtx        int      `json:"num_ctx,omitempty"`
	NumPredict    int      `json:"num_predict,omitempty"`
	Stop          []string `json:"stop,omitempty"`
	RepeatPenalty float64  `json:"repeat_penalty,omitempty"`
	RepeatLastN   int      `json:"repeat_last_n,omitempty"`
}

type GenerateResponse struct {
	Model     string  `json:"model"`
	Message   Message `json:"message"`
	Done      bool    `json:"done"`
	CreatedAt string  `json:"created_at"`

	TotalDuration      int64 `json:"total_duration,omitempty"`
	LoadDuration       int64 `json:"load_duration,omitempty"`
	PromptEvalCount    int   `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64 `json:"prompt_eval_duration,omitempty"`
	EvalCount          int   `json:"eval_count,omitempty"`
	EvalDuration       int64 `json:"eval_duration,omitempty"`
}

// --- Chat Methods ---

// Chat sends a message sequence and returns the full response (no tools).
func (c *Client) Chat(messages []Message, opts *ModelOptions) (*GenerateResponse, error) {
	return c.ChatWithTools(messages, nil, opts)
}

// ChatJSON sends a message sequence with format:"json" constraint.
// The model is forced to output only valid JSON — no free-form text.
func (c *Client) ChatJSON(messages []Message, opts *ModelOptions) (*GenerateResponse, error) {
	req := GenerateRequest{
		Model:    c.model,
		Messages: messages,
		Stream:   false,
		Options:  opts,
		Format:   "json",
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(c.host+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned %d: %s", resp.StatusCode, string(b))
	}

	var result GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}

// ChatWithTools sends a message sequence with tool definitions.
// When the model decides to call a tool, resp.Message.ToolCalls will be populated.
func (c *Client) ChatWithTools(messages []Message, tools []Tool, opts *ModelOptions) (*GenerateResponse, error) {
	req := GenerateRequest{
		Model:    c.model,
		Messages: messages,
		Stream:   false,
		Options:  opts,
		Tools:    tools,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(c.host+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned %d: %s", resp.StatusCode, string(b))
	}

	var result GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &result, nil
}

// StreamCallback is called for each token chunk during streaming.
type StreamCallback func(token string, done bool)

// ToolStreamCallback is called during streaming with native tool support.
// When the model calls a tool, toolCalls is non-nil and token is empty.
type ToolStreamCallback func(token string, toolCalls []ToolCall, done bool)

// ChatStream sends a message sequence and streams the response token by token.
func (c *Client) ChatStream(messages []Message, opts *ModelOptions, callback StreamCallback) (*GenerateResponse, error) {
	return c.ChatStreamWithTools(messages, nil, opts, func(token string, _ []ToolCall, done bool) {
		if callback != nil {
			callback(token, done)
		}
	})
}

// ChatStreamWithTools streams a response with native tool calling support.
func (c *Client) ChatStreamWithTools(messages []Message, tools []Tool, opts *ModelOptions, callback ToolStreamCallback) (*GenerateResponse, error) {
	req := GenerateRequest{
		Model:    c.model,
		Messages: messages,
		Stream:   true,
		Options:  opts,
		Tools:    tools,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	resp, err := c.httpClient.Post(c.host+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollama request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama returned %d: %s", resp.StatusCode, string(b))
	}

	var final GenerateResponse
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for scanner.Scan() {
		var chunk GenerateResponse
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			continue
		}

		if callback != nil {
			callback(chunk.Message.Content, chunk.Message.ToolCalls, chunk.Done)
		}

		if chunk.Done {
			final = chunk
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("stream read: %w", err)
	}

	return &final, nil
}

// --- Server Methods ---

// Ping checks if the Ollama server is reachable.
func (c *Client) Ping() error {
	resp, err := c.httpClient.Get(c.host + "/api/tags")
	if err != nil {
		return fmt.Errorf("ollama unreachable at %s: %w", c.host, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama returned status %d", resp.StatusCode)
	}
	return nil
}

// ModelInfo holds information about an available model.
type ModelInfo struct {
	Name       string `json:"name"`
	Size       int64  `json:"size"`
	ModifiedAt string `json:"modified_at"`
}

type listResponse struct {
	Models []ModelInfo `json:"models"`
}

// ListModels returns all models available in Ollama.
func (c *Client) ListModels() ([]ModelInfo, error) {
	resp, err := c.httpClient.Get(c.host + "/api/tags")
	if err != nil {
		return nil, fmt.Errorf("list models: %w", err)
	}
	defer resp.Body.Close()

	var result listResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode models: %w", err)
	}

	return result.Models, nil
}

// Model returns the currently configured model name.
func (c *Client) Model() string {
	return c.model
}

// Host returns the Ollama server address.
func (c *Client) Host() string {
	return c.host
}

// Clone creates a new client sharing the same host but targeting a different model.
func (c *Client) Clone(model string) *Client {
	return &Client{
		host:       c.host,
		model:      model,
		httpClient: c.httpClient,
	}
}

// --- Embeddings API ---

type EmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbedResponse struct {
	Embedding []float64 `json:"embedding"`
}

// Embed generates an embedding vector for the given text.
func (c *Client) Embed(text string) ([]float64, error) {
	req := EmbedRequest{
		Model:  c.model,
		Prompt: text,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal embed request: %w", err)
	}

	resp, err := c.httpClient.Post(c.host+"/api/embeddings", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("embed request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embed returned %d: %s", resp.StatusCode, string(b))
	}

	var result EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embed response: %w", err)
	}

	return result.Embedding, nil
}

// --- Create Model API ---

type CreateModelRequest struct {
	Name      string `json:"name"`
	Modelfile string `json:"modelfile"`
	Stream    bool   `json:"stream"`
}

type CreateModelResponse struct {
	Status string `json:"status"`
}

// CreateModel creates a new Ollama model from a Modelfile.
func (c *Client) CreateModel(name, modelfile string) error {
	req := CreateModelRequest{
		Name:      name,
		Modelfile: modelfile,
		Stream:    false,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal create request: %w", err)
	}

	resp, err := c.httpClient.Post(c.host+"/api/create", "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create model request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("create model returned %d: %s", resp.StatusCode, string(b))
	}

	return nil
}
