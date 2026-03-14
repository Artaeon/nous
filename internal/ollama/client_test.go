package ollama

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// helper: create a mock server that returns a chat response
func mockChatServer(t *testing.T, handler http.HandlerFunc) (*httptest.Server, *Client) {
	t.Helper()
	srv := httptest.NewServer(handler)
	client := New(WithHost(srv.URL))
	return srv, client
}

func chatOKHandler(content string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		resp := GenerateResponse{
			Model:   "test-model",
			Message: Message{Role: "assistant", Content: content},
			Done:    true,
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

// --- New() with options ---

func TestNewDefaults(t *testing.T) {
	c := New()
	if c.host != DefaultHost {
		t.Errorf("host = %q, want %q", c.host, DefaultHost)
	}
	if c.model != DefaultModel {
		t.Errorf("model = %q, want %q", c.model, DefaultModel)
	}
	if c.httpClient.Timeout != DefaultTimeout {
		t.Errorf("timeout = %v, want %v", c.httpClient.Timeout, DefaultTimeout)
	}
}

func TestNewWithHost(t *testing.T) {
	c := New(WithHost("http://example.com:1234"))
	if c.host != "http://example.com:1234" {
		t.Errorf("host = %q, want %q", c.host, "http://example.com:1234")
	}
}

func TestNewWithModel(t *testing.T) {
	c := New(WithModel("llama3:8b"))
	if c.model != "llama3:8b" {
		t.Errorf("model = %q, want %q", c.model, "llama3:8b")
	}
}

func TestNewWithTimeout(t *testing.T) {
	c := New(WithTimeout(10 * time.Second))
	if c.httpClient.Timeout != 10*time.Second {
		t.Errorf("timeout = %v, want %v", c.httpClient.Timeout, 10*time.Second)
	}
}

func TestNewMultipleOptions(t *testing.T) {
	c := New(WithHost("http://h:1"), WithModel("m"), WithTimeout(5*time.Second))
	if c.host != "http://h:1" {
		t.Errorf("host = %q", c.host)
	}
	if c.model != "m" {
		t.Errorf("model = %q", c.model)
	}
	if c.httpClient.Timeout != 5*time.Second {
		t.Errorf("timeout = %v", c.httpClient.Timeout)
	}
}

// --- Chat ---

func TestChat(t *testing.T) {
	srv, client := mockChatServer(t, chatOKHandler("hello world"))
	defer srv.Close()

	resp, err := client.Chat([]Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if resp.Message.Content != "hello world" {
		t.Errorf("Content = %q, want %q", resp.Message.Content, "hello world")
	}
}

func TestChatWithOptions(t *testing.T) {
	var receivedReq GenerateRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedReq)
		json.NewEncoder(w).Encode(GenerateResponse{Done: true})
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.Chat([]Message{{Role: "user", Content: "test"}}, &ModelOptions{Temperature: 0.5, NumPredict: 100})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if receivedReq.Options == nil {
		t.Fatal("Options should not be nil")
	}
	if receivedReq.Options.Temperature != 0.5 {
		t.Errorf("Temperature = %f, want 0.5", receivedReq.Options.Temperature)
	}
}

func TestChatServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.Chat([]Message{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "400") {
		t.Errorf("error = %q, want to contain '400'", err.Error())
	}
}

func TestChatConnectionError(t *testing.T) {
	client := New(WithHost("http://127.0.0.1:1")) // port 1 should refuse
	_, err := client.Chat([]Message{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("expected connection error")
	}
}

// --- ChatJSON ---

func TestChatJSON(t *testing.T) {
	var receivedReq GenerateRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedReq)
		resp := GenerateResponse{
			Message: Message{Role: "assistant", Content: `{"key":"value"}`},
			Done:    true,
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	resp, err := client.ChatJSON([]Message{{Role: "user", Content: "give json"}}, nil)
	if err != nil {
		t.Fatalf("ChatJSON error: %v", err)
	}
	if receivedReq.Format != "json" {
		t.Errorf("Format = %q, want %q", receivedReq.Format, "json")
	}
	if resp.Message.Content != `{"key":"value"}` {
		t.Errorf("Content = %q", resp.Message.Content)
	}
}

func TestChatJSONServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.ChatJSON([]Message{{Role: "user", Content: "test"}}, nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- ChatWithTools ---

func TestChatWithTools(t *testing.T) {
	var receivedReq GenerateRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedReq)
		resp := GenerateResponse{
			Message: Message{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					Function: ToolCallFunction{
						Name:      "search",
						Arguments: map[string]any{"query": "test"},
					},
				}},
			},
			Done: true,
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	tools := []Tool{{
		Type: "function",
		Function: ToolFunction{
			Name:        "search",
			Description: "Search for something",
			Parameters: ToolFunctionParams{
				Type: "object",
				Properties: map[string]ToolProperty{
					"query": {Type: "string", Description: "Search query"},
				},
				Required: []string{"query"},
			},
		},
	}}

	resp, err := client.ChatWithTools([]Message{{Role: "user", Content: "search for test"}}, tools, nil)
	if err != nil {
		t.Fatalf("ChatWithTools error: %v", err)
	}
	if len(receivedReq.Tools) != 1 {
		t.Fatalf("sent %d tools, want 1", len(receivedReq.Tools))
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("got %d tool calls, want 1", len(resp.Message.ToolCalls))
	}
	if resp.Message.ToolCalls[0].Function.Name != "search" {
		t.Errorf("tool call name = %q, want %q", resp.Message.ToolCalls[0].Function.Name, "search")
	}
}

// --- ChatStream ---

func TestChatStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chunks := []GenerateResponse{
			{Message: Message{Content: "hel"}, Done: false},
			{Message: Message{Content: "lo"}, Done: false},
			{Message: Message{Content: ""}, Done: true, EvalCount: 5},
		}
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "%s\n", data)
		}
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	var tokens []string
	var doneCount int

	resp, err := client.ChatStream(
		[]Message{{Role: "user", Content: "hi"}},
		nil,
		func(token string, done bool) {
			tokens = append(tokens, token)
			if done {
				doneCount++
			}
		},
	)
	if err != nil {
		t.Fatalf("ChatStream error: %v", err)
	}
	if len(tokens) != 3 {
		t.Errorf("got %d tokens, want 3", len(tokens))
	}
	if doneCount != 1 {
		t.Errorf("doneCount = %d, want 1", doneCount)
	}
	if resp.EvalCount != 5 {
		t.Errorf("EvalCount = %d, want 5", resp.EvalCount)
	}
}

func TestChatStreamNilCallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		chunk := GenerateResponse{Message: Message{Content: "ok"}, Done: true}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "%s\n", data)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.ChatStream([]Message{{Role: "user", Content: "hi"}}, nil, nil)
	if err != nil {
		t.Fatalf("ChatStream with nil callback error: %v", err)
	}
}

func TestChatStreamServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.ChatStream([]Message{{Role: "user", Content: "hi"}}, nil, nil)
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- ChatCtx ---

func TestChatCtx(t *testing.T) {
	srv, client := mockChatServer(t, chatOKHandler("context reply"))
	defer srv.Close()

	resp, err := client.ChatCtx(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("ChatCtx error: %v", err)
	}
	if resp.Message.Content != "context reply" {
		t.Errorf("Content = %q, want %q", resp.Message.Content, "context reply")
	}
}

func TestChatCtxCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		json.NewEncoder(w).Encode(GenerateResponse{Done: true})
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := client.ChatCtx(ctx, []Message{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestChatCtxServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "gone", http.StatusGone)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.ChatCtx(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "410") {
		t.Errorf("error = %q, want to contain '410'", err.Error())
	}
}

// --- Ping ---

func TestPing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Errorf("Ping hit path %q, want /api/tags", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"models":[]}`)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	if err := client.Ping(); err != nil {
		t.Errorf("Ping error: %v", err)
	}
}

func TestPingError(t *testing.T) {
	client := New(WithHost("http://127.0.0.1:1"))
	if err := client.Ping(); err == nil {
		t.Error("expected Ping error for unreachable host")
	}
}

func TestPingNonOK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	err := client.Ping()
	if err == nil {
		t.Error("expected error for non-200 status")
	}
	if !strings.Contains(err.Error(), "503") {
		t.Errorf("error = %q, want to contain '503'", err.Error())
	}
}

// --- ListModels ---

func TestListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := listResponse{
			Models: []ModelInfo{
				{Name: "llama3:8b", Size: 4000000000},
				{Name: "qwen2.5:1.5b", Size: 1500000000},
			},
		}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	models, err := client.ListModels()
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	if models[0].Name != "llama3:8b" {
		t.Errorf("models[0].Name = %q, want %q", models[0].Name, "llama3:8b")
	}
}

func TestListModelsEmpty(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(listResponse{Models: []ModelInfo{}})
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	models, err := client.ListModels()
	if err != nil {
		t.Fatalf("ListModels error: %v", err)
	}
	if len(models) != 0 {
		t.Errorf("got %d models, want 0", len(models))
	}
}

func TestListModelsError(t *testing.T) {
	client := New(WithHost("http://127.0.0.1:1"))
	_, err := client.ListModels()
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- Embed ---

func TestEmbed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embeddings" {
			t.Errorf("Embed hit path %q, want /api/embeddings", r.URL.Path)
		}
		var req EmbedRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Prompt != "hello" {
			t.Errorf("prompt = %q, want %q", req.Prompt, "hello")
		}
		json.NewEncoder(w).Encode(EmbedResponse{
			Embedding: []float64{0.1, 0.2, 0.3},
		})
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	emb, err := client.Embed("hello")
	if err != nil {
		t.Fatalf("Embed error: %v", err)
	}
	if len(emb) != 3 {
		t.Fatalf("got %d dims, want 3", len(emb))
	}
	if emb[0] != 0.1 {
		t.Errorf("emb[0] = %f, want 0.1", emb[0])
	}
}

func TestEmbedServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "fail", http.StatusInternalServerError)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.Embed("test")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("error = %q, want to contain '500'", err.Error())
	}
}

func TestEmbedConnectionError(t *testing.T) {
	client := New(WithHost("http://127.0.0.1:1"))
	_, err := client.Embed("test")
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- CreateModel ---

func TestCreateModel(t *testing.T) {
	var receivedReq CreateModelRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/create" {
			t.Errorf("CreateModel hit path %q, want /api/create", r.URL.Path)
		}
		json.NewDecoder(r.Body).Decode(&receivedReq)
		json.NewEncoder(w).Encode(CreateModelResponse{Status: "success"})
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	err := client.CreateModel("mymodel", "FROM llama3:8b\nSYSTEM You are helpful.")
	if err != nil {
		t.Fatalf("CreateModel error: %v", err)
	}
	if receivedReq.Name != "mymodel" {
		t.Errorf("Name = %q, want %q", receivedReq.Name, "mymodel")
	}
	if receivedReq.Modelfile != "FROM llama3:8b\nSYSTEM You are helpful." {
		t.Errorf("Modelfile = %q", receivedReq.Modelfile)
	}
}

func TestCreateModelServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	err := client.CreateModel("bad", "invalid")
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("error = %q, want to contain '404'", err.Error())
	}
}

// --- ensureNumCtx ---

func TestEnsureNumCtxNil(t *testing.T) {
	result := ensureNumCtx(nil)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if result.NumCtx != 8192 {
		t.Errorf("NumCtx = %d, want 8192", result.NumCtx)
	}
}

func TestEnsureNumCtxZero(t *testing.T) {
	opts := &ModelOptions{Temperature: 0.7}
	result := ensureNumCtx(opts)
	if result.NumCtx != 8192 {
		t.Errorf("NumCtx = %d, want 8192", result.NumCtx)
	}
	// Should not mutate the original
	if opts.NumCtx != 0 {
		t.Error("original opts was mutated")
	}
	// Should preserve other fields
	if result.Temperature != 0.7 {
		t.Errorf("Temperature = %f, want 0.7", result.Temperature)
	}
}

func TestEnsureNumCtxAlreadySet(t *testing.T) {
	opts := &ModelOptions{NumCtx: 4096}
	result := ensureNumCtx(opts)
	if result != opts {
		t.Error("expected same pointer when NumCtx already set")
	}
	if result.NumCtx != 4096 {
		t.Errorf("NumCtx = %d, want 4096", result.NumCtx)
	}
}

// --- Clone ---

func TestClone(t *testing.T) {
	original := New(WithHost("http://myhost:1234"), WithModel("llama3:8b"))
	cloned := original.Clone("qwen2.5:1.5b")

	if cloned.host != "http://myhost:1234" {
		t.Errorf("cloned host = %q, want %q", cloned.host, "http://myhost:1234")
	}
	if cloned.model != "qwen2.5:1.5b" {
		t.Errorf("cloned model = %q, want %q", cloned.model, "qwen2.5:1.5b")
	}
	if cloned.httpClient != original.httpClient {
		t.Error("cloned should share httpClient")
	}
	// Original should be unchanged
	if original.model != "llama3:8b" {
		t.Errorf("original model = %q, want %q", original.model, "llama3:8b")
	}
}

// --- Model / Host ---

func TestModelAndHost(t *testing.T) {
	c := New(WithHost("http://h:1"), WithModel("m"))
	if c.Model() != "m" {
		t.Errorf("Model() = %q, want %q", c.Model(), "m")
	}
	if c.Host() != "http://h:1" {
		t.Errorf("Host() = %q, want %q", c.Host(), "http://h:1")
	}
}

// --- ToolResultMessage ---

func TestToolResultMessage(t *testing.T) {
	msg := ToolResultMessage("search", "found 3 results")
	if msg.Role != "tool" {
		t.Errorf("Role = %q, want %q", msg.Role, "tool")
	}
	if msg.ToolName != "search" {
		t.Errorf("ToolName = %q, want %q", msg.ToolName, "search")
	}
	if msg.Content != "found 3 results" {
		t.Errorf("Content = %q, want %q", msg.Content, "found 3 results")
	}
}

// --- Invalid JSON response ---

func TestChatInvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "not json at all")
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.Chat([]Message{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("expected error for invalid JSON response")
	}
}

func TestEmbedInvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "{{bad")
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.Embed("test")
	if err == nil {
		t.Fatal("expected error for invalid JSON response")
	}
}

func TestListModelsInvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "not json")
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL))
	_, err := client.ListModels()
	if err == nil {
		t.Fatal("expected error for invalid JSON response")
	}
}

// --- Verify request body structure ---

func TestChatRequestStructure(t *testing.T) {
	var receivedReq GenerateRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&receivedReq)
		json.NewEncoder(w).Encode(GenerateResponse{Done: true})
	}))
	defer srv.Close()

	client := New(WithHost(srv.URL), WithModel("test-model"))
	_, err := client.Chat([]Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
	}, nil)
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	if receivedReq.Model != "test-model" {
		t.Errorf("Model = %q, want %q", receivedReq.Model, "test-model")
	}
	if receivedReq.Stream != false {
		t.Error("Stream should be false for Chat")
	}
	if len(receivedReq.Messages) != 2 {
		t.Errorf("Messages count = %d, want 2", len(receivedReq.Messages))
	}
	if receivedReq.KeepAlive != "30m" {
		t.Errorf("KeepAlive = %q, want %q", receivedReq.KeepAlive, "30m")
	}
}
