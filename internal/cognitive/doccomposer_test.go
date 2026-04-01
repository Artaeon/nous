package cognitive

import (
	"strings"
	"testing"
)

func TestDocComposer_DraftEmail_Formal(t *testing.T) {
	dc := NewDocComposer()
	result := dc.DraftEmail("Alice", "project deadline", "formal", []string{
		"The deadline has been moved to Friday.",
		"Please submit your review by Wednesday.",
	})

	if result == "" {
		t.Fatal("expected non-empty email")
	}
	if !strings.Contains(result, "Alice") {
		t.Error("email should address recipient")
	}
	if !strings.Contains(result, "project deadline") {
		t.Error("email should mention subject")
	}
	if !strings.Contains(result, "Friday") {
		t.Error("email should contain fact about Friday")
	}
	// Formal email should NOT have casual greetings.
	if strings.Contains(result, "Hey Alice") {
		t.Error("formal email should not use 'Hey'")
	}
}

func TestDocComposer_DraftEmail_Casual(t *testing.T) {
	dc := NewDocComposer()
	result := dc.DraftEmail("Bob", "lunch plans", "casual", []string{
		"Let's try the new place on Main Street.",
	})

	if result == "" {
		t.Fatal("expected non-empty email")
	}
	if !strings.Contains(result, "Bob") {
		t.Error("email should address recipient")
	}
	if !strings.Contains(result, "lunch plans") {
		t.Error("email should mention subject")
	}
	// Casual should not have "Dear".
	if strings.Contains(result, "Dear Bob") {
		t.Error("casual email should not use 'Dear'")
	}
}

func TestDocComposer_DraftEmail_Urgent(t *testing.T) {
	dc := NewDocComposer()
	result := dc.DraftEmail("Team", "server outage", "urgent", []string{
		"Production server is down since 14:00.",
		"Customer-facing APIs are affected.",
	})

	if result == "" {
		t.Fatal("expected non-empty email")
	}
	lower := strings.ToLower(result)
	if !strings.Contains(lower, "urgent") && !strings.Contains(lower, "immediate") {
		t.Error("urgent email should convey urgency")
	}
}

func TestDocComposer_DraftEmail_EmptyParams(t *testing.T) {
	dc := NewDocComposer()
	result := dc.Draft(DocParams{Type: "email"})

	if result == "" {
		t.Fatal("email with empty params should still produce structure")
	}
	// Should have some default greeting.
	if !strings.Contains(result, "there") {
		t.Error("email with no recipient should default to 'there'")
	}
}

func TestDocComposer_DraftReport(t *testing.T) {
	dc := NewDocComposer()
	result := dc.DraftReport("Q1 Review", nil, []string{
		"Revenue up 15%.",
		"Team grew to 8.",
		"Shipped v2.0.",
	})

	if result == "" {
		t.Fatal("expected non-empty report")
	}
	if !strings.Contains(result, "Q1 Review") {
		t.Error("report should contain title")
	}
	if !strings.Contains(result, "## Executive Summary") {
		t.Error("report should have Executive Summary section")
	}
	if !strings.Contains(result, "## Findings") {
		t.Error("report should have Findings section")
	}
	if !strings.Contains(result, "Revenue up 15%") {
		t.Error("report should include facts")
	}
	if !strings.Contains(result, "## Recommendations") {
		t.Error("report should have Recommendations section")
	}
	if !strings.Contains(result, "## Next Steps") {
		t.Error("report should have Next Steps section")
	}
}

func TestDocComposer_DraftReport_EmptyFacts(t *testing.T) {
	dc := NewDocComposer()
	result := dc.DraftReport("Empty Report", nil, nil)

	if result == "" {
		t.Fatal("report with no facts should still produce structure")
	}
	if !strings.Contains(result, "Empty Report") {
		t.Error("report should contain title")
	}
	if !strings.Contains(result, "## Executive Summary") {
		t.Error("report should still have sections even with no facts")
	}
}

func TestDocComposer_DraftMeetingNotes(t *testing.T) {
	dc := NewDocComposer()
	result := dc.DraftMeetingNotes(
		[]string{"Alice", "Bob", "Carol"},
		[]string{"Ship v2 by end of month", "Hire two more engineers"},
		[]string{"Alice: write docs", "Bob: deploy staging", "Carol: review PRs"},
	)

	if result == "" {
		t.Fatal("expected non-empty meeting notes")
	}
	if !strings.Contains(result, "# Meeting Notes") {
		t.Error("meeting notes should have header")
	}
	if !strings.Contains(result, "## Attendees") {
		t.Error("meeting notes should have Attendees section")
	}
	if !strings.Contains(result, "Alice") {
		t.Error("meeting notes should list attendees")
	}
	if !strings.Contains(result, "Bob") {
		t.Error("meeting notes should list attendees")
	}
	if !strings.Contains(result, "## Decisions") {
		t.Error("meeting notes should have Decisions section")
	}
	if !strings.Contains(result, "Ship v2") {
		t.Error("meeting notes should include decisions")
	}
	if !strings.Contains(result, "## Action Items") {
		t.Error("meeting notes should have Action Items section")
	}
	if !strings.Contains(result, "[ ]") {
		t.Error("action items should have checkboxes")
	}
	if !strings.Contains(result, "Alice: write docs") {
		t.Error("action items should contain assigned tasks")
	}
}

func TestDocComposer_DraftMeetingNotes_Empty(t *testing.T) {
	dc := NewDocComposer()
	result := dc.DraftMeetingNotes(nil, nil, nil)

	if result == "" {
		t.Fatal("meeting notes with no params should still produce structure")
	}
	if !strings.Contains(result, "# Meeting Notes") {
		t.Error("should have header even with no params")
	}
	if !strings.Contains(result, "## Attendees") {
		t.Error("should have Attendees section")
	}
}

func TestDocComposer_DraftProposal(t *testing.T) {
	dc := NewDocComposer()
	result := dc.Draft(DocParams{
		Type:    "proposal",
		Subject: "New CI Pipeline",
		Facts: []string{
			"Current builds take 45 minutes.",
			"Proposed pipeline reduces to 10 minutes using caching.",
			"Estimated cost: $200/month for cloud runners.",
		},
		Actions: []string{
			"Evaluate GitHub Actions vs GitLab CI",
			"Set up proof of concept",
			"Present results to team",
		},
	})

	if result == "" {
		t.Fatal("expected non-empty proposal")
	}
	if !strings.Contains(result, "New CI Pipeline") {
		t.Error("proposal should contain title")
	}
	if !strings.Contains(result, "## Executive Summary") {
		t.Error("proposal should have Executive Summary")
	}
	if !strings.Contains(result, "## Problem Statement") {
		t.Error("proposal should have Problem Statement")
	}
	if !strings.Contains(result, "## Proposed Solution") {
		t.Error("proposal should have Proposed Solution")
	}
	if !strings.Contains(result, "## Benefits") {
		t.Error("proposal should have Benefits")
	}
	if !strings.Contains(result, "## Next Steps") {
		t.Error("proposal should have Next Steps")
	}
	if !strings.Contains(result, "Evaluate GitHub Actions") {
		t.Error("proposal should include action items")
	}
}

func TestDocComposer_DraftStatus(t *testing.T) {
	dc := NewDocComposer()
	result := dc.Draft(DocParams{
		Type: "status",
		Facts: []string{
			"Finished database migration.",
			"Working on API refactor.",
			"Blocked on security review.",
			"Plan to start frontend work next week.",
		},
	})

	if result == "" {
		t.Fatal("expected non-empty status update")
	}
	if !strings.Contains(result, "# Status Update") {
		t.Error("status should have header")
	}
	if !strings.Contains(result, "## Completed") {
		t.Error("status should have Completed section")
	}
	if !strings.Contains(result, "## In Progress") {
		t.Error("status should have In Progress section")
	}
	if !strings.Contains(result, "## Blocked") {
		t.Error("status should have Blocked section")
	}
	if !strings.Contains(result, "## Next Steps") {
		t.Error("status should have Next Steps section")
	}
	// Heuristic categorization: "Finished" -> completed, "Working on" -> in progress, "Blocked" -> blocked.
	if !strings.Contains(result, "database migration") {
		t.Error("status should include all facts")
	}
}

func TestDocComposer_DraftStatus_Empty(t *testing.T) {
	dc := NewDocComposer()
	result := dc.Draft(DocParams{Type: "status"})

	if result == "" {
		t.Fatal("status with no facts should still produce structure")
	}
	if !strings.Contains(result, "# Status Update") {
		t.Error("should have header even with no facts")
	}
}

func TestDocComposer_SupportedTypes(t *testing.T) {
	dc := NewDocComposer()
	types := dc.SupportedTypes()

	if len(types) < 5 {
		t.Errorf("expected at least 5 supported types, got %d", len(types))
	}

	// Check that all core types are present.
	typeSet := make(map[string]bool)
	for _, typ := range types {
		typeSet[typ] = true
	}
	for _, expected := range []string{"email", "report", "meeting-notes", "proposal", "status"} {
		if !typeSet[expected] {
			t.Errorf("expected supported type %q not found", expected)
		}
	}
}

func TestDocComposer_AllTypesProduceOutput(t *testing.T) {
	dc := NewDocComposer()
	testCases := []DocParams{
		{Type: "email", To: "Test", Subject: "test", Facts: []string{"point one"}},
		{Type: "report", Subject: "test report", Facts: []string{"finding one"}},
		{Type: "meeting-notes", Attendees: []string{"A"}, Actions: []string{"do thing"}},
		{Type: "proposal", Subject: "test proposal", Facts: []string{"problem"}},
		{Type: "status", Facts: []string{"done something"}},
	}

	for _, tc := range testCases {
		result := dc.Draft(tc)
		if result == "" {
			t.Errorf("Draft(%q) returned empty string", tc.Type)
		}
	}
}

func TestDocComposer_UnknownType_FallsBackToGeneric(t *testing.T) {
	dc := NewDocComposer()
	result := dc.Draft(DocParams{
		Type:    "unknown-type",
		Subject: "Miscellaneous",
		Facts:   []string{"item one", "item two"},
	})

	if result == "" {
		t.Fatal("unknown type should produce generic output")
	}
	if !strings.Contains(result, "Miscellaneous") {
		t.Error("generic output should contain subject")
	}
}

func TestDocComposer_StatusCategorization(t *testing.T) {
	dc := NewDocComposer()

	// Test heuristic categorization.
	params := DocParams{
		Facts: []string{
			"Shipped the new landing page.",
			"Working on API integration.",
			"Blocked on third-party credentials.",
			"Next: start testing phase.",
		},
	}
	completed, inProgress, blocked, planned := dc.categorizeFacts(params)

	if len(completed) == 0 {
		t.Error("expected 'Shipped' fact to be categorized as completed")
	}
	if len(inProgress) == 0 {
		t.Error("expected 'Working on' fact to be categorized as in-progress")
	}
	if len(blocked) == 0 {
		t.Error("expected 'Blocked' fact to be categorized as blocked")
	}
	if len(planned) == 0 {
		t.Error("expected remaining fact to be categorized as planned")
	}
}

func TestDocComposer_DraftEmail_SignoffContainsFrom(t *testing.T) {
	dc := NewDocComposer()
	result := dc.Draft(DocParams{
		Type:    "email",
		To:      "Alice",
		From:    "Bob",
		Subject: "hello",
		Tone:    "formal",
		Facts:   []string{"Just saying hello."},
	})

	if !strings.Contains(result, "Bob") {
		t.Error("email should contain sender name in sign-off")
	}
}

func TestDocComposer_MeetingNotesVariantTypes(t *testing.T) {
	dc := NewDocComposer()

	// Test that "meeting_notes" and "meetingnotes" also work.
	for _, typ := range []string{"meeting-notes", "meeting_notes", "meetingnotes"} {
		result := dc.Draft(DocParams{
			Type:      typ,
			Attendees: []string{"X"},
		})
		if !strings.Contains(result, "# Meeting Notes") {
			t.Errorf("type %q should produce meeting notes", typ)
		}
	}
}
