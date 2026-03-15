package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"

	"github.com/artaeon/nous/internal/ollama"
)

// NewOllamaClientFromOpenAI creates an ollama.Client that is backed by an
// OpenAI-compatible API. This allows the existing cognitive components
// (which all take *ollama.Client) to work with OpenAI, Groq, Together,
// vLLM, or any OpenAI-compatible endpoint without refactoring.
//
// It works by running a local HTTP translation proxy that converts
// Ollama API calls into OpenAI API calls and translates responses back.
func NewOllamaClientFromOpenAI(baseURL, apiKey, model string) (*ollama.Client, func()) {
	proxy := &openaiProxy{
		baseURL:    baseURL,
		apiKey:     apiKey,
		httpClient: &http.Client{},
	}

	server := httptest.NewServer(proxy)

	client := ollama.New(
		ollama.WithHost(server.URL),
		ollama.WithModel(model),
	)

	cleanup := func() {
		server.Close()
	}

	return client, cleanup
}

// openaiProxy translates Ollama API requests to OpenAI API requests.
type openaiProxy struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	mu         sync.Mutex
}

func (p *openaiProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path

	switch {
	case path == "/api/tags":
		// Ping / list models — return a minimal valid response
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{},
		})

	case path == "/api/chat":
		p.handleChat(w, r)

	case path == "/api/embeddings":
		p.handleEmbed(w, r)

	default:
		http.Error(w, "not found", http.StatusNotFound)
	}
}

func (p *openaiProxy) handleChat(w http.ResponseWriter, r *http.Request) {
	var ollamaReq ollama.GenerateRequest
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := json.Unmarshal(body, &ollamaReq); err != nil {
		http.Error(w, "parse body: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Convert Ollama messages to OpenAI format
	oaiMessages := make([]openaiMessage, len(ollamaReq.Messages))
	for i, m := range ollamaReq.Messages {
		role := m.Role
		// OpenAI doesn't have a "tool" role in the same way;
		// map it to "user" with a prefix for compatibility
		if role == "tool" {
			role = "user"
			if m.ToolName != "" {
				oaiMessages[i] = openaiMessage{
					Role:    role,
					Content: fmt.Sprintf("[Tool result from %s]: %s", m.ToolName, m.Content),
				}
				continue
			}
		}
		oaiMessages[i] = openaiMessage{Role: role, Content: m.Content}
	}

	oaiReq := openaiRequest{
		Model:    ollamaReq.Model,
		Messages: oaiMessages,
		Stream:   false, // always non-streaming through proxy
	}

	if ollamaReq.Options != nil {
		if ollamaReq.Options.Temperature > 0 {
			t := ollamaReq.Options.Temperature
			oaiReq.Temperature = &t
		}
		if ollamaReq.Options.TopP > 0 {
			t := ollamaReq.Options.TopP
			oaiReq.TopP = &t
		}
		if ollamaReq.Options.NumPredict > 0 {
			m := ollamaReq.Options.NumPredict
			oaiReq.MaxTokens = &m
		}
		if len(ollamaReq.Options.Stop) > 0 {
			oaiReq.Stop = ollamaReq.Options.Stop
		}
	}

	if fmt.Sprintf("%v", ollamaReq.Format) == "json" {
		// OpenAI supports response_format for JSON mode, but we handle
		// it by adding an instruction since not all compatible APIs support it.
		// The model should already be instructed to output JSON.
	}

	oaiBody, err := json.Marshal(oaiReq)
	if err != nil {
		http.Error(w, "marshal openai request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", p.baseURL+"/chat/completions", bytes.NewReader(oaiBody))
	if err != nil {
		http.Error(w, "create request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		http.Error(w, "openai request failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "read openai response: "+err.Error(), http.StatusBadGateway)
		return
	}

	if resp.StatusCode != http.StatusOK {
		http.Error(w, string(respBody), resp.StatusCode)
		return
	}

	var oaiResp openaiResponse
	if err := json.Unmarshal(respBody, &oaiResp); err != nil {
		http.Error(w, "decode openai response: "+err.Error(), http.StatusBadGateway)
		return
	}

	if oaiResp.Error != nil {
		http.Error(w, oaiResp.Error.Message, http.StatusBadGateway)
		return
	}

	content := ""
	if len(oaiResp.Choices) > 0 {
		content = oaiResp.Choices[0].Message.Content
	}

	// Build Ollama-format response
	ollamaResp := ollama.GenerateResponse{
		Model: ollamaReq.Model,
		Message: ollama.Message{
			Role:    "assistant",
			Content: content,
		},
		Done:            true,
		PromptEvalCount: oaiResp.Usage.PromptTokens,
		EvalCount:       oaiResp.Usage.CompletionTokens,
	}

	// For streaming requests, send as newline-delimited JSON (Ollama format)
	if ollamaReq.Stream {
		w.Header().Set("Content-Type", "application/x-ndjson")

		// Send the content as a single chunk then a done message
		if content != "" {
			chunk := ollama.GenerateResponse{
				Model: ollamaReq.Model,
				Message: ollama.Message{
					Role:    "assistant",
					Content: content,
				},
				Done: false,
			}
			chunkJSON, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "%s\n", chunkJSON)
		}

		// Send done
		doneResp := ollamaResp
		doneResp.Message.Content = ""
		doneJSON, _ := json.Marshal(doneResp)
		fmt.Fprintf(w, "%s\n", doneJSON)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ollamaResp)
}

func (p *openaiProxy) handleEmbed(w http.ResponseWriter, r *http.Request) {
	var ollamaReq ollama.EmbedRequest
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := json.Unmarshal(body, &ollamaReq); err != nil {
		http.Error(w, "parse body: "+err.Error(), http.StatusBadRequest)
		return
	}

	oaiReq := openaiEmbedRequest{
		Model: ollamaReq.Model,
		Input: ollamaReq.Prompt,
	}

	oaiBody, err := json.Marshal(oaiReq)
	if err != nil {
		http.Error(w, "marshal embed request: "+err.Error(), http.StatusInternalServerError)
		return
	}

	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", p.baseURL+"/embeddings", bytes.NewReader(oaiBody))
	if err != nil {
		http.Error(w, "create embed request: "+err.Error(), http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		http.Error(w, "embed request failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "read embed response: "+err.Error(), http.StatusBadGateway)
		return
	}

	if resp.StatusCode != http.StatusOK {
		http.Error(w, string(respBody), resp.StatusCode)
		return
	}

	var oaiResp openaiEmbedResponse
	if err := json.Unmarshal(respBody, &oaiResp); err != nil {
		http.Error(w, "decode embed response: "+err.Error(), http.StatusBadGateway)
		return
	}

	embedding := []float64{}
	if len(oaiResp.Data) > 0 {
		embedding = oaiResp.Data[0].Embedding
	}

	ollamaResp := ollama.EmbedResponse{
		Embedding: embedding,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ollamaResp)
}
