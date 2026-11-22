package chattemplate

import (
	"encoding/json"
	"testing"
)

func TestTupleMappingMembership(t *testing.T) {
	tmpl, err := Parse(`{{ ('prefix', items) in lookup }}`, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := tmpl.Render(Input{Messages: oneUserMessage, Kwargs: json.RawMessage(`{"items":[],"lookup":{}}`)})
	if err == nil || err.Error() != "TypeError: unhashable type: 'list'" {
		t.Fatalf("Render = (%q, %v); want an unhashable list error", got, err)
	}
}
