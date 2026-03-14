package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// okHandler is a trivial handler that returns 200 with {"ok":true}.
var okHandler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
})

func TestAuthMiddleware_ValidBearerToken(t *testing.T) {
	handler := AuthMiddleware("secret-key", okHandler)

	req := httptest.NewRequest("GET", "/api/status", nil)
	req.Header.Set("Authorization", "Bearer secret-key")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rr.Code)
	}
}

func TestAuthMiddleware_InvalidBearerToken(t *testing.T) {
	handler := AuthMiddleware("secret-key", okHandler)

	req := httptest.NewRequest("GET", "/api/status", nil)
	req.Header.Set("Authorization", "Bearer wrong-key")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}

	var body map[string]string
	json.NewDecoder(rr.Body).Decode(&body)
	if body["error"] != "unauthorized" {
		t.Errorf("expected error 'unauthorized', got %q", body["error"])
	}
}

func TestAuthMiddleware_MissingKey(t *testing.T) {
	handler := AuthMiddleware("secret-key", okHandler)

	req := httptest.NewRequest("GET", "/api/status", nil)
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rr.Code)
	}
}

func TestAuthMiddleware_HealthBypass(t *testing.T) {
	handler := AuthMiddleware("secret-key", okHandler)

	req := httptest.NewRequest("GET", "/api/health", nil)
	// No auth header — should still pass.
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 for health bypass, got %d", rr.Code)
	}
}

func TestAuthMiddleware_QueryParamAuth(t *testing.T) {
	handler := AuthMiddleware("secret-key", okHandler)

	req := httptest.NewRequest("GET", "/api/status?api_key=secret-key", nil)
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 with query param auth, got %d", rr.Code)
	}
}

func TestAuthMiddleware_QueryParamWrongKey(t *testing.T) {
	handler := AuthMiddleware("secret-key", okHandler)

	req := httptest.NewRequest("GET", "/api/status?api_key=nope", nil)
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 with wrong query param, got %d", rr.Code)
	}
}

func TestAuthMiddleware_EmptyKeySkipsAuth(t *testing.T) {
	handler := AuthMiddleware("", okHandler)

	req := httptest.NewRequest("GET", "/api/status", nil)
	// No auth at all — should pass because key is empty (local mode).
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 when no key configured, got %d", rr.Code)
	}
}

func TestAuthMiddleware_CaseInsensitiveBearer(t *testing.T) {
	handler := AuthMiddleware("my-key", okHandler)

	req := httptest.NewRequest("GET", "/api/status", nil)
	req.Header.Set("Authorization", "bearer my-key")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 with case-insensitive bearer, got %d", rr.Code)
	}
}
