package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOllamaAdapter_Chat(t *testing.T) {
	// Spin up a fake Ollama server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.Error(w, "not found", 404)
			return
		}

		resp := map[string]any{
			"model": "test-model",
			"message": map[string]string{
				"role":    "assistant",
				"content": "Hello from Ollama!",
			},
			"done":              true,
			"prompt_eval_count": 5,
			"eval_count":        8,
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	// Create real Ollama client pointing at fake server
	ollamaClient := newTestOllamaClient(server.URL, "test-model")
	adapter := NewOllama(ollamaClient)

	// Verify it satisfies Provider interface
	var _ Provider = adapter

	resp, err := adapter.Chat([]Message{
		{Role: "user", Content: "hello"},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "Hello from Ollama!" {
		t.Errorf("got content %q, want %q", resp.Content, "Hello from Ollama!")
	}
	if resp.TokensUsed != 13 {
		t.Errorf("got tokens %d, want %d", resp.TokensUsed, 13)
	}
}

func TestOllamaAdapter_ChatCtx(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"model": "test-model",
			"message": map[string]string{
				"role":    "assistant",
				"content": "ctx response",
			},
			"done": true,
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	ollamaClient := newTestOllamaClient(server.URL, "test-model")
	adapter := NewOllama(ollamaClient)

	resp, err := adapter.ChatCtx(context.Background(), []Message{
		{Role: "user", Content: "hi"},
	}, &Options{Temperature: 0.5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "ctx response" {
		t.Errorf("got %q, want %q", resp.Content, "ctx response")
	}
}

func TestOllamaAdapter_Embed(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embeddings" {
			http.Error(w, "not found", 404)
			return
		}
		resp := map[string]any{
			"embedding": []float64{0.1, 0.2, 0.3},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	ollamaClient := newTestOllamaClient(server.URL, "test-model")
	adapter := NewOllama(ollamaClient)

	vec, err := adapter.Embed("test text")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vec) != 3 {
		t.Errorf("got %d dimensions, want 3", len(vec))
	}
}

func TestOllamaAdapter_Model(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"models": []any{}})
	}))
	defer server.Close()

	ollamaClient := newTestOllamaClient(server.URL, "my-model")
	adapter := NewOllama(ollamaClient)

	if adapter.Model() != "my-model" {
		t.Errorf("got %q, want %q", adapter.Model(), "my-model")
	}
}

func TestOllamaAdapter_Client(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"models": []any{}})
	}))
	defer server.Close()

	ollamaClient := newTestOllamaClient(server.URL, "test-model")
	adapter := NewOllama(ollamaClient)

	if adapter.Client() != ollamaClient {
		t.Error("Client() should return the underlying ollama.Client")
	}
}

func TestConvertOptions(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		if convertOptions(nil) != nil {
			t.Error("nil input should return nil")
		}
	})

	t.Run("all fields", func(t *testing.T) {
		opts := &Options{
			Temperature: 0.7,
			TopP:        0.9,
			MaxTokens:   512,
			NumCtx:      4096,
			Stop:        []string{"END"},
		}
		result := convertOptions(opts)
		if result.Temperature != 0.7 {
			t.Errorf("temperature: got %f, want 0.7", result.Temperature)
		}
		if result.TopP != 0.9 {
			t.Errorf("top_p: got %f, want 0.9", result.TopP)
		}
		if result.NumPredict != 512 {
			t.Errorf("num_predict: got %d, want 512", result.NumPredict)
		}
		if result.NumCtx != 4096 {
			t.Errorf("num_ctx: got %d, want 4096", result.NumCtx)
		}
		if len(result.Stop) != 1 || result.Stop[0] != "END" {
			t.Errorf("stop: got %v, want [END]", result.Stop)
		}
	})
}
