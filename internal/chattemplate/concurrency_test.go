package chattemplate

import (
	"sync"
	"testing"
)

// One Template serves every request of a server, so rendering must not share
// state between calls. Run with -race.
func TestRenderIsSafeForConcurrentUse(t *testing.T) {
	const goroutines = 16
	for _, family := range goldenFamilies {
		t.Run(family.name, func(t *testing.T) {
			tmpl := loadVendoredTemplate(t, family.name)
			rows := readRows[requestRow](t, goldenPath(family.name), goldenRowCount)
			var wg sync.WaitGroup
			for range goroutines {
				wg.Go(func() {
					for _, row := range rows {
						got, err := tmpl.Render(row.input())
						if mismatch := row.check(got, err); mismatch != nil {
							t.Errorf("%s: %v", row.Name, mismatch)
						}
					}
				})
			}
			wg.Wait()
		})
	}
}
