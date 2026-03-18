package server

import (
	"encoding/json"
	"net/http"
	"strings"
)

// AuthMiddleware returns an http.Handler that validates API key authentication.
// Requests must include either an "Authorization: Bearer <key>" header or an
// "api_key" query parameter matching the configured key.
//
// The /api/health endpoint is always allowed without authentication so that
// external monitoring can probe liveness.
//
// If apiKey is empty, all requests are passed through without checks (local mode).
func AuthMiddleware(apiKey string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// No key configured — local mode, skip auth entirely.
		if apiKey == "" {
			next.ServeHTTP(w, r)
			return
		}

		// Health endpoint is exempt so load balancers and monitors can probe it.
		// Root path serves the web UI login page which must load before auth.
		if r.URL.Path == "/api/health" || r.URL.Path == "/" {
			next.ServeHTTP(w, r)
			return
		}

		// Check Authorization header first.
		if extractBearerToken(r) == apiKey {
			next.ServeHTTP(w, r)
			return
		}

		// Fallback: query parameter.
		if r.URL.Query().Get("api_key") == apiKey {
			next.ServeHTTP(w, r)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]string{"error": "unauthorized"})
	})
}

// extractBearerToken reads the bearer token from the Authorization header.
func extractBearerToken(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if auth == "" {
		return ""
	}
	const prefix = "Bearer "
	if len(auth) > len(prefix) && strings.EqualFold(auth[:len(prefix)], prefix) {
		return auth[len(prefix):]
	}
	return ""
}
