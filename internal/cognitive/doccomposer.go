package cognitive

import (
	"fmt"
	"math/rand"
	"strings"
	"time"
)

// DocComposer generates structured documents from intent and parameters.
// No ML required — uses document-type-specific templates with slot filling.
type DocComposer struct {
	templates map[string]*DocTemplate
	rng       *rand.Rand
}

// DocTemplate describes a document type with ordered sections.
type DocTemplate struct {
	Type     string       // "email", "report", "meeting-notes", etc.
	Sections []DocSection
}

// DocSection is one section of a document template.
type DocSection struct {
	Name     string   // "greeting", "body", "closing", etc.
	Required bool
	Variants []string // template variants with {slot} placeholders
}

// DocParams captures the intent and content for document generation.
type DocParams struct {
	Type      string            // document type
	To        string            // recipient
	From      string            // sender (optional)
	Subject   string            // topic/about
	Tone      string            // formal, casual, friendly, urgent
	Facts     []string          // key points to include
	Sections  []string          // requested sections (for reports)
	Attendees []string          // for meeting notes
	Decisions []string          // for meeting notes
	Actions   []string          // action items
	Extra     map[string]string // additional parameters
}

// NewDocComposer creates a DocComposer initialized with all built-in templates.
func NewDocComposer() *DocComposer {
	dc := &DocComposer{
		templates: make(map[string]*DocTemplate),
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
	dc.registerEmail()
	dc.registerReport()
	dc.registerMeetingNotes()
	dc.registerProposal()
	dc.registerStatus()
	return dc
}

// SupportedTypes returns the names of all registered document types.
func (dc *DocComposer) SupportedTypes() []string {
	types := make([]string, 0, len(dc.templates))
	for k := range dc.templates {
		types = append(types, k)
	}
	return types
}

// Draft generates a document from the given parameters.
func (dc *DocComposer) Draft(params DocParams) string {
	switch strings.ToLower(params.Type) {
	case "email":
		return dc.draftEmail(params)
	case "report":
		return dc.draftReport(params)
	case "meeting-notes", "meeting_notes", "meetingnotes":
		return dc.draftMeetingNotes(params)
	case "proposal":
		return dc.draftProposal(params)
	case "status", "status-update", "status_update":
		return dc.draftStatus(params)
	default:
		return dc.draftGeneric(params)
	}
}

// DraftEmail is a convenience method for email generation.
func (dc *DocComposer) DraftEmail(to, subject, tone string, points []string) string {
	return dc.Draft(DocParams{
		Type:    "email",
		To:      to,
		Subject: subject,
		Tone:    tone,
		Facts:   points,
	})
}

// DraftReport is a convenience method for report generation.
func (dc *DocComposer) DraftReport(title string, sections []string, facts []string) string {
	return dc.Draft(DocParams{
		Type:     "report",
		Subject:  title,
		Sections: sections,
		Facts:    facts,
	})
}

// DraftMeetingNotes is a convenience method for meeting notes generation.
func (dc *DocComposer) DraftMeetingNotes(attendees, decisions, actions []string) string {
	return dc.Draft(DocParams{
		Type:      "meeting-notes",
		Attendees: attendees,
		Decisions: decisions,
		Actions:   actions,
	})
}

// ---------------------------------------------------------------------------
// Template registration
// ---------------------------------------------------------------------------

func (dc *DocComposer) registerEmail() {
	dc.templates["email"] = &DocTemplate{
		Type: "email",
		Sections: []DocSection{
			{
				Name:     "greeting",
				Required: true,
				Variants: []string{
					"Dear {to},",
					"Hello {to},",
					"Good {timeofday} {to},",
				},
			},
			{
				Name:     "opening",
				Required: true,
				Variants: []string{
					"I am writing to you regarding {subject}.",
					"I wanted to reach out about {subject}.",
					"I hope this message finds you well. I am writing regarding {subject}.",
					"Thank you for your time. I would like to discuss {subject}.",
				},
			},
			{
				Name:     "body",
				Required: true,
				Variants: []string{
					"{facts}",
				},
			},
			{
				Name:     "closing",
				Required: true,
				Variants: []string{
					"Please let me know if you have any questions or need further information.",
					"I look forward to your response at your earliest convenience.",
					"Please do not hesitate to reach out if you need any clarification.",
					"I would appreciate your feedback on this matter.",
				},
			},
			{
				Name:     "signoff",
				Required: true,
				Variants: []string{
					"Best regards,\n{from}",
					"Sincerely,\n{from}",
					"Kind regards,\n{from}",
				},
			},
		},
	}

	dc.templates["email-casual"] = &DocTemplate{
		Type: "email",
		Sections: []DocSection{
			{
				Name:     "greeting",
				Required: true,
				Variants: []string{
					"Hi {to},",
					"Hey {to},",
					"Hi there {to},",
				},
			},
			{
				Name:     "opening",
				Required: true,
				Variants: []string{
					"Just wanted to touch base about {subject}.",
					"Quick note about {subject}.",
					"Hope you're doing well! Wanted to chat about {subject}.",
					"Reaching out about {subject}.",
				},
			},
			{
				Name:     "body",
				Required: true,
				Variants: []string{
					"{facts}",
				},
			},
			{
				Name:     "closing",
				Required: true,
				Variants: []string{
					"Let me know what you think!",
					"Happy to chat more about this whenever works for you.",
					"Would love to hear your thoughts.",
					"Ping me if you have questions!",
				},
			},
			{
				Name:     "signoff",
				Required: true,
				Variants: []string{
					"Cheers,\n{from}",
					"Thanks,\n{from}",
					"Talk soon,\n{from}",
					"Best,\n{from}",
				},
			},
		},
	}

	dc.templates["email-urgent"] = &DocTemplate{
		Type: "email",
		Sections: []DocSection{
			{
				Name:     "greeting",
				Required: true,
				Variants: []string{
					"Dear {to},",
					"{to},",
				},
			},
			{
				Name:     "opening",
				Required: true,
				Variants: []string{
					"This is an urgent matter regarding {subject} that requires your immediate attention.",
					"I need to bring {subject} to your attention immediately.",
					"Urgent: please review the following regarding {subject}.",
				},
			},
			{
				Name:     "body",
				Required: true,
				Variants: []string{
					"{facts}",
				},
			},
			{
				Name:     "closing",
				Required: true,
				Variants: []string{
					"Please respond as soon as possible.",
					"This requires action today. Please confirm receipt.",
					"Immediate response requested.",
				},
			},
			{
				Name:     "signoff",
				Required: true,
				Variants: []string{
					"Regards,\n{from}",
					"Best regards,\n{from}",
				},
			},
		},
	}
}

func (dc *DocComposer) registerReport() {
	dc.templates["report"] = &DocTemplate{
		Type: "report",
		Sections: []DocSection{
			{Name: "title", Required: true},
			{Name: "executive-summary", Required: true},
			{Name: "background", Required: false},
			{Name: "findings", Required: true},
			{Name: "analysis", Required: false},
			{Name: "recommendations", Required: false},
			{Name: "next-steps", Required: false},
		},
	}
}

func (dc *DocComposer) registerMeetingNotes() {
	dc.templates["meeting-notes"] = &DocTemplate{
		Type: "meeting-notes",
		Sections: []DocSection{
			{Name: "header", Required: true},
			{Name: "attendees", Required: true},
			{Name: "discussion", Required: false},
			{Name: "decisions", Required: false},
			{Name: "action-items", Required: true},
		},
	}
}

func (dc *DocComposer) registerProposal() {
	dc.templates["proposal"] = &DocTemplate{
		Type: "proposal",
		Sections: []DocSection{
			{Name: "executive-summary", Required: true},
			{Name: "problem-statement", Required: true},
			{Name: "proposed-solution", Required: true},
			{Name: "benefits", Required: false},
			{Name: "timeline", Required: false},
			{Name: "resources", Required: false},
			{Name: "next-steps", Required: false},
		},
	}
}

func (dc *DocComposer) registerStatus() {
	dc.templates["status"] = &DocTemplate{
		Type: "status",
		Sections: []DocSection{
			{Name: "header", Required: true},
			{Name: "completed", Required: false},
			{Name: "in-progress", Required: false},
			{Name: "blocked", Required: false},
			{Name: "next-steps", Required: false},
		},
	}
}

// ---------------------------------------------------------------------------
// Document drafters
// ---------------------------------------------------------------------------

func (dc *DocComposer) draftEmail(params DocParams) string {
	tone := strings.ToLower(params.Tone)
	templateKey := "email"
	switch tone {
	case "casual", "friendly":
		templateKey = "email-casual"
	case "urgent":
		templateKey = "email-urgent"
	}

	tmpl := dc.templates[templateKey]
	if tmpl == nil {
		tmpl = dc.templates["email"]
	}

	slots := dc.buildSlots(params)
	var b strings.Builder

	for _, sec := range tmpl.Sections {
		if len(sec.Variants) == 0 {
			continue
		}
		variant := sec.Variants[dc.rng.Intn(len(sec.Variants))]
		line := dc.fillSlots(variant, slots)
		b.WriteString(line)
		b.WriteString("\n\n")
	}

	return strings.TrimSpace(b.String())
}

func (dc *DocComposer) draftReport(params DocParams) string {
	slots := dc.buildSlots(params)
	var b strings.Builder

	// Title
	title := params.Subject
	if title == "" {
		title = "Report"
	}
	b.WriteString("# " + title + "\n\n")

	// Executive Summary
	b.WriteString("## Executive Summary\n\n")
	if len(params.Facts) > 0 {
		// First 1-2 facts as the summary.
		limit := 2
		if len(params.Facts) < limit {
			limit = len(params.Facts)
		}
		b.WriteString(strings.Join(params.Facts[:limit], " "))
		b.WriteString("\n\n")
	} else {
		b.WriteString("This report covers " + dc.fillSlots("{subject}", slots) + ".\n\n")
	}

	// Background/Context
	b.WriteString("## Background\n\n")
	if extra, ok := params.Extra["background"]; ok && extra != "" {
		b.WriteString(extra + "\n\n")
	} else {
		b.WriteString("This section provides context for " + dc.fillSlots("{subject}", slots) + ".\n\n")
	}

	// Findings
	if len(params.Facts) > 0 {
		b.WriteString("## Findings\n\n")
		for i, fact := range params.Facts {
			b.WriteString(fmt.Sprintf("%d. %s\n", i+1, fact))
		}
		b.WriteString("\n")
	}

	// Analysis
	b.WriteString("## Analysis\n\n")
	if len(params.Facts) > 0 {
		b.WriteString("The key findings indicate the following trends:\n\n")
		for _, fact := range params.Facts {
			b.WriteString("- " + fact + "\n")
		}
		b.WriteString("\n")
	} else {
		b.WriteString("Further analysis is needed.\n\n")
	}

	// Recommendations
	b.WriteString("## Recommendations\n\n")
	if len(params.Actions) > 0 {
		for _, action := range params.Actions {
			b.WriteString("- " + action + "\n")
		}
		b.WriteString("\n")
	} else if len(params.Facts) > 0 {
		b.WriteString("Based on the findings, the following steps are recommended:\n\n")
		for i, fact := range params.Facts {
			b.WriteString(fmt.Sprintf("%d. Review and address: %s\n", i+1, fact))
		}
		b.WriteString("\n")
	} else {
		b.WriteString("Recommendations will follow after further analysis.\n\n")
	}

	// Next Steps
	b.WriteString("## Next Steps\n\n")
	if len(params.Actions) > 0 {
		for _, action := range params.Actions {
			b.WriteString("- [ ] " + action + "\n")
		}
	} else {
		b.WriteString("- [ ] Review this report with stakeholders\n")
		b.WriteString("- [ ] Identify action owners\n")
		b.WriteString("- [ ] Schedule follow-up\n")
	}

	return strings.TrimSpace(b.String())
}

func (dc *DocComposer) draftMeetingNotes(params DocParams) string {
	dateStr := dc.currentDate()

	var b strings.Builder

	// Header
	b.WriteString("# Meeting Notes — " + dateStr + "\n\n")

	// Attendees
	b.WriteString("## Attendees\n\n")
	if len(params.Attendees) > 0 {
		for _, a := range params.Attendees {
			b.WriteString("- " + strings.TrimSpace(a) + "\n")
		}
	} else {
		b.WriteString("- (no attendees listed)\n")
	}
	b.WriteString("\n")

	// Discussion
	b.WriteString("## Discussion\n\n")
	if len(params.Facts) > 0 {
		for _, fact := range params.Facts {
			b.WriteString("- " + fact + "\n")
		}
	} else if params.Subject != "" {
		b.WriteString("- " + params.Subject + "\n")
	} else {
		b.WriteString("- (no discussion items recorded)\n")
	}
	b.WriteString("\n")

	// Decisions
	b.WriteString("## Decisions\n\n")
	if len(params.Decisions) > 0 {
		for _, d := range params.Decisions {
			b.WriteString("- " + strings.TrimSpace(d) + "\n")
		}
	} else {
		b.WriteString("- (no decisions recorded)\n")
	}
	b.WriteString("\n")

	// Action Items
	b.WriteString("## Action Items\n\n")
	if len(params.Actions) > 0 {
		for _, a := range params.Actions {
			b.WriteString("- [ ] " + strings.TrimSpace(a) + "\n")
		}
	} else {
		b.WriteString("- [ ] (no action items recorded)\n")
	}

	return strings.TrimSpace(b.String())
}

func (dc *DocComposer) draftProposal(params DocParams) string {
	var b strings.Builder

	title := params.Subject
	if title == "" {
		title = "Proposal"
	}
	b.WriteString("# " + title + "\n\n")

	// Executive Summary
	b.WriteString("## Executive Summary\n\n")
	if len(params.Facts) > 0 {
		b.WriteString(strings.Join(params.Facts, " ") + "\n\n")
	} else {
		b.WriteString("This proposal addresses " + title + ".\n\n")
	}

	// Problem Statement
	b.WriteString("## Problem Statement\n\n")
	if extra, ok := params.Extra["problem"]; ok && extra != "" {
		b.WriteString(extra + "\n\n")
	} else if len(params.Facts) > 0 {
		b.WriteString(params.Facts[0] + "\n\n")
	} else {
		b.WriteString("The problem to be addressed is described below.\n\n")
	}

	// Proposed Solution
	b.WriteString("## Proposed Solution\n\n")
	if extra, ok := params.Extra["solution"]; ok && extra != "" {
		b.WriteString(extra + "\n\n")
	} else if len(params.Facts) > 1 {
		b.WriteString(strings.Join(params.Facts[1:], " ") + "\n\n")
	} else {
		b.WriteString("The proposed approach is outlined below.\n\n")
	}

	// Benefits
	b.WriteString("## Benefits\n\n")
	if extra, ok := params.Extra["benefits"]; ok && extra != "" {
		for _, benefit := range strings.Split(extra, ".") {
			benefit = strings.TrimSpace(benefit)
			if benefit != "" {
				b.WriteString("- " + benefit + "\n")
			}
		}
		b.WriteString("\n")
	} else {
		b.WriteString("- Addresses the identified problem directly\n")
		b.WriteString("- Provides a clear path forward\n\n")
	}

	// Timeline
	b.WriteString("## Timeline\n\n")
	if extra, ok := params.Extra["timeline"]; ok && extra != "" {
		b.WriteString(extra + "\n\n")
	} else {
		b.WriteString("Timeline to be determined based on resource availability.\n\n")
	}

	// Resources Required
	b.WriteString("## Resources Required\n\n")
	if extra, ok := params.Extra["resources"]; ok && extra != "" {
		b.WriteString(extra + "\n\n")
	} else {
		b.WriteString("Resource requirements to be assessed during planning phase.\n\n")
	}

	// Next Steps
	b.WriteString("## Next Steps\n\n")
	if len(params.Actions) > 0 {
		for _, a := range params.Actions {
			b.WriteString("- [ ] " + strings.TrimSpace(a) + "\n")
		}
	} else {
		b.WriteString("- [ ] Review proposal with stakeholders\n")
		b.WriteString("- [ ] Gather feedback\n")
		b.WriteString("- [ ] Refine timeline and resource estimates\n")
	}

	return strings.TrimSpace(b.String())
}

func (dc *DocComposer) draftStatus(params DocParams) string {
	dateStr := dc.currentDate()

	var b strings.Builder
	b.WriteString("# Status Update — " + dateStr + "\n\n")

	// Categorize facts into completed, in-progress, blocked, planned.
	completed, inProgress, blocked, planned := dc.categorizeFacts(params)

	// Completed
	b.WriteString("## Completed\n\n")
	if len(completed) > 0 {
		for _, item := range completed {
			b.WriteString("- " + item + "\n")
		}
	} else {
		b.WriteString("- (none)\n")
	}
	b.WriteString("\n")

	// In Progress
	b.WriteString("## In Progress\n\n")
	if len(inProgress) > 0 {
		for _, item := range inProgress {
			b.WriteString("- " + item + "\n")
		}
	} else {
		b.WriteString("- (none)\n")
	}
	b.WriteString("\n")

	// Blocked
	b.WriteString("## Blocked\n\n")
	if len(blocked) > 0 {
		for _, item := range blocked {
			b.WriteString("- " + item + "\n")
		}
	} else {
		b.WriteString("- (none)\n")
	}
	b.WriteString("\n")

	// Next Steps
	b.WriteString("## Next Steps\n\n")
	if len(planned) > 0 {
		for _, item := range planned {
			b.WriteString("- " + item + "\n")
		}
	} else if len(params.Actions) > 0 {
		for _, a := range params.Actions {
			b.WriteString("- " + strings.TrimSpace(a) + "\n")
		}
	} else {
		b.WriteString("- (none)\n")
	}

	return strings.TrimSpace(b.String())
}

func (dc *DocComposer) draftGeneric(params DocParams) string {
	var b strings.Builder

	title := params.Subject
	if title == "" {
		title = "Document"
	}
	b.WriteString("# " + title + "\n\n")

	if len(params.Facts) > 0 {
		for _, fact := range params.Facts {
			b.WriteString("- " + fact + "\n")
		}
	}

	if len(params.Actions) > 0 {
		b.WriteString("\n## Action Items\n\n")
		for _, a := range params.Actions {
			b.WriteString("- [ ] " + strings.TrimSpace(a) + "\n")
		}
	}

	return strings.TrimSpace(b.String())
}

// ---------------------------------------------------------------------------
// Slot filling and helpers
// ---------------------------------------------------------------------------

func (dc *DocComposer) buildSlots(params DocParams) map[string]string {
	slots := map[string]string{
		"to":        params.To,
		"from":      params.From,
		"subject":   params.Subject,
		"date":      dc.currentDate(),
		"timeofday": dc.timeOfDay(),
	}

	if slots["to"] == "" {
		slots["to"] = "there"
	}
	if slots["from"] == "" {
		slots["from"] = ""
	}
	if slots["subject"] == "" {
		slots["subject"] = "the matter at hand"
	}

	// Format facts as a bulleted list or paragraph.
	if len(params.Facts) > 0 {
		if len(params.Facts) == 1 {
			slots["facts"] = params.Facts[0]
		} else {
			var lines []string
			for _, f := range params.Facts {
				lines = append(lines, "- "+f)
			}
			slots["facts"] = strings.Join(lines, "\n")
		}
	} else {
		slots["facts"] = ""
	}

	// Copy extra params into slots.
	for k, v := range params.Extra {
		slots[k] = v
	}

	return slots
}

func (dc *DocComposer) fillSlots(template string, slots map[string]string) string {
	result := template
	for k, v := range slots {
		result = strings.ReplaceAll(result, "{"+k+"}", v)
	}
	return result
}

func (dc *DocComposer) currentDate() string {
	return time.Now().Format("2006-01-02")
}

func (dc *DocComposer) timeOfDay() string {
	hour := time.Now().Hour()
	switch {
	case hour < 12:
		return "morning"
	case hour < 17:
		return "afternoon"
	default:
		return "evening"
	}
}

// categorizeFacts splits facts into completed, in-progress, blocked, and planned
// based on keyword heuristics. Facts in Extra["completed"], Extra["in-progress"],
// Extra["blocked"], and params.Actions are used directly when available.
func (dc *DocComposer) categorizeFacts(params DocParams) (completed, inProgress, blocked, planned []string) {
	// Check for explicit categorization in Extra.
	if extra, ok := params.Extra["completed"]; ok && extra != "" {
		completed = splitItems(extra)
	}
	if extra, ok := params.Extra["in-progress"]; ok && extra != "" {
		inProgress = splitItems(extra)
	}
	if extra, ok := params.Extra["blocked"]; ok && extra != "" {
		blocked = splitItems(extra)
	}
	if extra, ok := params.Extra["planned"]; ok && extra != "" {
		planned = splitItems(extra)
	}

	// If explicit categories were provided, done.
	if len(completed)+len(inProgress)+len(blocked)+len(planned) > 0 {
		return
	}

	// Otherwise, heuristically categorize facts by keywords.
	for _, fact := range params.Facts {
		lower := strings.ToLower(fact)
		switch {
		case strings.Contains(lower, "done") || strings.Contains(lower, "completed") ||
			strings.Contains(lower, "finished") || strings.Contains(lower, "shipped"):
			completed = append(completed, fact)
		case strings.Contains(lower, "blocked") || strings.Contains(lower, "stuck") ||
			strings.Contains(lower, "waiting") || strings.Contains(lower, "blocker"):
			blocked = append(blocked, fact)
		case strings.Contains(lower, "working on") || strings.Contains(lower, "in progress") ||
			strings.Contains(lower, "started") || strings.Contains(lower, "ongoing"):
			inProgress = append(inProgress, fact)
		default:
			// Default: treat as planned/next.
			planned = append(planned, fact)
		}
	}

	// Actions go to planned if not already placed.
	planned = append(planned, params.Actions...)
	return
}

// splitItems splits a sentence-boundary or comma separated string into trimmed
// items. Splits on ". " (period + space) to avoid breaking decimals like "v2.0".
func splitItems(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}

	var items []string
	remaining := s
	for {
		idx := strings.Index(remaining, ". ")
		if idx < 0 {
			seg := strings.TrimSpace(strings.TrimRight(remaining, "."))
			if seg != "" {
				items = append(items, seg)
			}
			break
		}
		seg := strings.TrimSpace(remaining[:idx])
		if seg != "" {
			items = append(items, seg)
		}
		remaining = remaining[idx+2:]
	}

	if len(items) > 1 {
		return items
	}

	// Fall back to comma.
	items = nil
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			items = append(items, part)
		}
	}
	return items
}
