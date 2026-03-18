package tools

import (
	"crypto/md5"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"net/url"
	"strings"
)

// HashEncodeDecode performs hashing, encoding, and decoding operations.
// Supported actions: md5, sha1, sha256, sha512, base64encode, base64decode,
// urlencode, urldecode, hex, unhex.
// Default action is sha256 if not specified.
func HashEncodeDecode(action, input string) (string, error) {
	if input == "" {
		return "", fmt.Errorf("hash: input is required")
	}

	action = strings.ToLower(strings.TrimSpace(action))
	if action == "" {
		action = "sha256"
	}

	switch action {
	case "md5":
		h := md5.Sum([]byte(input))
		return fmt.Sprintf("md5: %x", h), nil
	case "sha1":
		h := sha1.Sum([]byte(input))
		return fmt.Sprintf("sha1: %x", h), nil
	case "sha256":
		h := sha256.Sum256([]byte(input))
		return fmt.Sprintf("sha256: %x", h), nil
	case "sha512":
		h := sha512.Sum512([]byte(input))
		return fmt.Sprintf("sha512: %x", h), nil
	case "base64encode":
		encoded := base64.StdEncoding.EncodeToString([]byte(input))
		return fmt.Sprintf("base64encode: %s", encoded), nil
	case "base64decode":
		decoded, err := base64.StdEncoding.DecodeString(input)
		if err != nil {
			return "", fmt.Errorf("hash: base64 decode failed: %w", err)
		}
		return fmt.Sprintf("base64decode: %s", string(decoded)), nil
	case "urlencode":
		encoded := url.QueryEscape(input)
		return fmt.Sprintf("urlencode: %s", encoded), nil
	case "urldecode":
		decoded, err := url.QueryUnescape(input)
		if err != nil {
			return "", fmt.Errorf("hash: url decode failed: %w", err)
		}
		return fmt.Sprintf("urldecode: %s", decoded), nil
	case "hex":
		encoded := hex.EncodeToString([]byte(input))
		return fmt.Sprintf("hex: %s", encoded), nil
	case "unhex":
		decoded, err := hex.DecodeString(input)
		if err != nil {
			return "", fmt.Errorf("hash: hex decode failed: %w", err)
		}
		return fmt.Sprintf("unhex: %s", string(decoded)), nil
	default:
		return "", fmt.Errorf("hash: unknown action %q — supported: md5, sha1, sha256, sha512, base64encode, base64decode, urlencode, urldecode, hex, unhex", action)
	}
}

// RegisterHashTools adds the hash tool to the registry.
func RegisterHashTools(r *Registry) {
	r.Register(Tool{
		Name:        "hash",
		Description: "Hash, encode, or decode data. Args: action (md5/sha1/sha256/sha512/base64encode/base64decode/urlencode/urldecode/hex/unhex, default sha256), input (required).",
		Execute: func(args map[string]string) (string, error) {
			return HashEncodeDecode(args["action"], args["input"])
		},
	})
}
