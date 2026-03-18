package tools

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// EmailConfig holds IMAP connection settings.
type EmailConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Username string `json:"username"`
	Password string `json:"password"`
	Mailbox  string `json:"mailbox"`
	UseTLS   bool   `json:"use_tls"`
}

// EmailMessage represents a fetched email header.
type EmailMessage struct {
	From    string
	Subject string
	Date    time.Time
}

// LoadEmailConfig reads email configuration from ~/.nous/email.json.
func LoadEmailConfig() (*EmailConfig, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, fmt.Errorf("email: cannot determine home directory: %w", err)
	}

	path := filepath.Join(home, ".nous", "email.json")
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // signal no config
		}
		return nil, fmt.Errorf("email: %w", err)
	}

	var config EmailConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("email: invalid config: %w", err)
	}

	// Defaults
	if config.Port == 0 {
		if config.UseTLS {
			config.Port = 993
		} else {
			config.Port = 143
		}
	}
	if config.Mailbox == "" {
		config.Mailbox = "INBOX"
	}

	return &config, nil
}

// CheckEmail connects via IMAP and fetches unread email summaries.
func CheckEmail(config *EmailConfig, maxMessages int) (string, error) {
	if config == nil {
		return emailSetupMessage(), nil
	}
	if maxMessages <= 0 {
		maxMessages = 5
	}

	addr := fmt.Sprintf("%s:%d", config.Host, config.Port)

	var conn net.Conn
	var err error

	if config.UseTLS {
		conn, err = tls.DialWithDialer(
			&net.Dialer{Timeout: 10 * time.Second},
			"tcp", addr,
			&tls.Config{ServerName: config.Host},
		)
	} else {
		conn, err = net.DialTimeout("tcp", addr, 10*time.Second)
	}
	if err != nil {
		return "", fmt.Errorf("email: connect to %s: %w", addr, err)
	}
	defer conn.Close()

	conn.SetDeadline(time.Now().Add(30 * time.Second))

	// Helper to read a response line
	readLine := func() (string, error) {
		var buf [4096]byte
		n, err := conn.Read(buf[:])
		if err != nil {
			return "", err
		}
		return string(buf[:n]), nil
	}

	// Helper to send a command and read response
	tagN := 0
	sendCmd := func(cmd string) (string, error) {
		tagN++
		tag := fmt.Sprintf("A%03d", tagN)
		full := fmt.Sprintf("%s %s\r\n", tag, cmd)
		if _, err := conn.Write([]byte(full)); err != nil {
			return "", fmt.Errorf("email: send: %w", err)
		}

		var response strings.Builder
		for {
			line, err := readLine()
			if err != nil {
				return response.String(), fmt.Errorf("email: read: %w", err)
			}
			response.WriteString(line)
			if strings.Contains(line, tag+" OK") || strings.Contains(line, tag+" NO") || strings.Contains(line, tag+" BAD") {
				break
			}
		}
		resp := response.String()
		if strings.Contains(resp, tag+" NO") || strings.Contains(resp, tag+" BAD") {
			return resp, fmt.Errorf("email: command failed: %s", strings.TrimSpace(resp))
		}
		return resp, nil
	}

	// Read server greeting
	if _, err := readLine(); err != nil {
		return "", fmt.Errorf("email: greeting: %w", err)
	}

	// LOGIN
	if _, err := sendCmd(fmt.Sprintf("LOGIN %s %s", config.Username, config.Password)); err != nil {
		return "", fmt.Errorf("email: login failed: %w", err)
	}

	// SELECT mailbox
	if _, err := sendCmd(fmt.Sprintf("SELECT %s", config.Mailbox)); err != nil {
		return "", fmt.Errorf("email: select %s: %w", config.Mailbox, err)
	}

	// SEARCH UNSEEN
	searchResp, err := sendCmd("SEARCH UNSEEN")
	if err != nil {
		return "", fmt.Errorf("email: search: %w", err)
	}

	// Parse message IDs from "* SEARCH 1 2 3"
	var msgIDs []string
	for _, line := range strings.Split(searchResp, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "* SEARCH") {
			parts := strings.Fields(line)
			if len(parts) > 2 {
				msgIDs = parts[2:]
			}
		}
	}

	if len(msgIDs) == 0 {
		sendCmd("LOGOUT")
		return "No unread emails.", nil
	}

	totalUnread := len(msgIDs)

	// Limit to last N messages
	if len(msgIDs) > maxMessages {
		msgIDs = msgIDs[len(msgIDs)-maxMessages:]
	}

	// FETCH envelopes
	var messages []EmailMessage
	idList := strings.Join(msgIDs, ",")
	fetchResp, err := sendCmd(fmt.Sprintf("FETCH %s (ENVELOPE)", idList))
	if err != nil {
		// Still try to show count even if fetch fails
		sendCmd("LOGOUT")
		return fmt.Sprintf("You have %d unread email(s) (could not fetch details).", totalUnread), nil
	}

	// Parse envelope responses (simplified)
	messages = parseEnvelopeResponses(fetchResp)

	// LOGOUT
	sendCmd("LOGOUT")

	return FormatEmailSummary(messages, totalUnread), nil
}

// parseEnvelopeResponses does a best-effort parse of IMAP FETCH ENVELOPE responses.
func parseEnvelopeResponses(resp string) []EmailMessage {
	var messages []EmailMessage

	lines := strings.Split(resp, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, "ENVELOPE") {
			continue
		}

		var msg EmailMessage

		// Extract subject: find the subject field in the envelope
		// Envelope format: (date subject from sender reply-to to cc bcc in-reply-to message-id)
		envStart := strings.Index(line, "ENVELOPE (")
		if envStart < 0 {
			continue
		}
		envData := line[envStart+10:]

		// Parse date (first quoted string)
		msg.Date = parseEnvelopeDate(envData)

		// Parse subject (second quoted string)
		msg.Subject = extractNthQuoted(envData, 2)

		// Parse from (third field, which is a nested list)
		msg.From = extractFromField(envData)

		if msg.Subject != "" || msg.From != "" {
			messages = append(messages, msg)
		}
	}

	return messages
}

// extractNthQuoted extracts the Nth quoted string from IMAP data.
func extractNthQuoted(s string, n int) string {
	count := 0
	i := 0
	for i < len(s) {
		if s[i] == '"' {
			count++
			end := strings.Index(s[i+1:], "\"")
			if end < 0 {
				break
			}
			if count == n {
				return s[i+1 : i+1+end]
			}
			i = i + 1 + end + 1
		} else {
			i++
		}
	}
	return ""
}

// parseEnvelopeDate parses the date from an IMAP envelope.
func parseEnvelopeDate(s string) time.Time {
	dateStr := extractNthQuoted(s, 1)
	if dateStr == "" {
		return time.Time{}
	}

	formats := []string{
		"Mon, 2 Jan 2006 15:04:05 -0700",
		"Mon, 02 Jan 2006 15:04:05 -0700",
		"2 Jan 2006 15:04:05 -0700",
		"Mon, 2 Jan 2006 15:04:05 MST",
	}

	for _, fmt := range formats {
		if t, err := time.Parse(fmt, dateStr); err == nil {
			return t
		}
	}
	return time.Time{}
}

// extractFromField does a best-effort extraction of the sender name/email from envelope data.
func extractFromField(s string) string {
	// The from field is the 3rd parenthesized list in the envelope
	// Look for pattern like ((NIL NIL "user" "domain.com"))
	// or (("Display Name" NIL "user" "domain.com"))
	depth := 0
	listCount := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '(' {
			depth++
			if depth == 1 {
				listCount++
			}
			if listCount == 3 && depth >= 2 {
				// We're inside the from field's inner list
				end := strings.Index(s[i:], "))")
				if end < 0 {
					break
				}
				inner := s[i : i+end+2]
				// Try to extract "user" "domain"
				parts := extractAllQuoted(inner)
				if len(parts) >= 4 {
					email := parts[2] + "@" + parts[3]
					if parts[0] != "" && parts[0] != "NIL" {
						return parts[0] + " <" + email + ">"
					}
					return email
				}
				if len(parts) >= 1 && parts[0] != "" {
					return parts[0]
				}
				break
			}
		} else if s[i] == ')' {
			depth--
		}
	}
	return ""
}

// extractAllQuoted returns all quoted strings from a string.
func extractAllQuoted(s string) []string {
	var result []string
	i := 0
	for i < len(s) {
		if s[i] == '"' {
			end := strings.Index(s[i+1:], "\"")
			if end < 0 {
				break
			}
			result = append(result, s[i+1:i+1+end])
			i = i + 1 + end + 1
		} else if i+3 <= len(s) && s[i:i+3] == "NIL" {
			result = append(result, "")
			i += 3
		} else {
			i++
		}
	}
	return result
}

// FormatEmailSummary formats email messages for display.
func FormatEmailSummary(messages []EmailMessage, totalUnread int) string {
	if totalUnread == 0 {
		return "No unread emails."
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("You have %d unread email(s):\n", totalUnread))

	for i, msg := range messages {
		from := msg.From
		if from == "" {
			from = "(unknown sender)"
		}
		subject := msg.Subject
		if subject == "" {
			subject = "(no subject)"
		}

		age := ""
		if !msg.Date.IsZero() {
			age = " (" + formatTimeAgo(msg.Date) + ")"
		}

		fmt.Fprintf(&sb, "%d. From: %s - Subject: %s%s\n", i+1, from, subject, age)
	}

	return strings.TrimRight(sb.String(), "\n")
}

// formatTimeAgo returns a human-readable time difference.
func formatTimeAgo(t time.Time) string {
	diff := time.Since(t)
	switch {
	case diff < time.Minute:
		return "just now"
	case diff < time.Hour:
		m := int(diff.Minutes())
		if m == 1 {
			return "1m ago"
		}
		return fmt.Sprintf("%dm ago", m)
	case diff < 24*time.Hour:
		h := int(diff.Hours())
		if h == 1 {
			return "1h ago"
		}
		return fmt.Sprintf("%dh ago", h)
	default:
		d := int(diff.Hours() / 24)
		if d == 1 {
			return "1d ago"
		}
		return fmt.Sprintf("%dd ago", d)
	}
}

// emailSetupMessage returns instructions for setting up email checking.
func emailSetupMessage() string {
	home, _ := os.UserHomeDir()
	configPath := filepath.Join(home, ".nous", "email.json")
	return fmt.Sprintf(`Email not configured. Create %s with:
{
  "host": "imap.example.com",
  "port": 993,
  "username": "you@example.com",
  "password": "your-app-password",
  "use_tls": true
}`, configPath)
}

// RegisterEmailTools adds the email tool to the registry.
func RegisterEmailTools(r *Registry) {
	r.Register(Tool{
		Name:        "email",
		Description: "Check for unread emails via IMAP. Args: count (optional, default 5).",
		Execute: func(args map[string]string) (string, error) {
			config, err := LoadEmailConfig()
			if err != nil {
				return "", err
			}
			maxMessages := 5
			if v, ok := args["count"]; ok {
				if n, err := strconv.Atoi(v); err == nil && n > 0 {
					maxMessages = n
				}
			}
			return CheckEmail(config, maxMessages)
		},
	})
}
