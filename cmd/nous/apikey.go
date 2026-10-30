package main

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// loadAPIKey keeps file-based credentials out of process arguments and Docker
// metadata. Errors intentionally contain neither the key nor the file path.
func loadAPIKey(value, path string) (string, error) {
	if path == "" {
		return value, nil
	}
	if value != "" {
		return "", fmt.Errorf("choose either --api-key or --api-key-file")
	}
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0077 != 0 {
		return "", fmt.Errorf("API key file must be a readable regular file with owner-only permissions")
	}
	f, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("cannot read API key file")
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil || !os.SameFile(info, opened) || opened.Mode().Perm()&0077 != 0 {
		return "", fmt.Errorf("API key file changed while opening")
	}
	data, err := io.ReadAll(io.LimitReader(f, 4097))
	key := strings.TrimSpace(string(data))
	if err != nil || len(data) > 4096 || len(key) < 32 || strings.ContainsAny(key, " \t\r\n") {
		return "", fmt.Errorf("API key file must contain a single key of at least 32 characters, at most 4096 bytes")
	}
	return key, nil
}
