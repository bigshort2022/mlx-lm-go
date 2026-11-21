package mlx

import (
	"errors"
	"testing"
)

func TestScopedPropagatesOtherPanics(t *testing.T) {
	defer func() {
		if got := recover(); got != "boom" {
			t.Fatalf("recovered %v, want the original panic value \"boom\"", got)
		}
	}()
	_ = Scoped(func() error { panic("boom") })
	t.Fatal("Scoped returned, want it to panic")
}

func TestScopedReturnsFnError(t *testing.T) {
	want := errors.New("from fn")
	if got := Scoped(func() error { return want }); got != want {
		t.Fatalf("Scoped returned %v, want the error fn returned", got)
	}
}

func TestScopedFreesArraysCreatedInside(t *testing.T) {
	pin(t)
	var a, b *Array
	err := Scoped(func() error {
		a = FromFloat32([]float32{1, 2}, 2)
		b = Matmul(a, a)
		if !a.Valid() || !b.Valid() {
			t.Error("arrays are invalid inside their scope")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if a.Valid() || b.Valid() {
		t.Fatalf("after Scoped: a.Valid()=%v b.Valid()=%v, want both false", a.Valid(), b.Valid())
	}
}

func TestScopedFreesArraysWhenAnOperationFails(t *testing.T) {
	pin(t)
	var a *Array
	err := Scoped(func() error {
		a = FromFloat32([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		Matmul(a, a)
		t.Error("Matmul of [2,3] x [2,3] returned, want it to panic")
		return nil
	})
	if err == nil {
		t.Error("Scoped returned nil, want the Matmul error")
	}
	if a.Valid() {
		t.Error("array is still valid after its scope ended with an error")
	}
}

func TestScopedClosesItsScopeWhenFnPanics(t *testing.T) {
	pin(t)
	var inside *Array
	func() {
		defer func() { recover() }()
		_ = Scoped(func() error {
			inside = FromFloat32([]float32{1, 2}, 2)
			panic("boom")
		})
	}()
	if inside.Valid() {
		t.Error("array is still valid after its scope ended with a panic")
	}
	after := FromFloat32([]float32{1, 2}, 2)
	defer after.Free()
	if err := Scoped(func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if !after.Valid() {
		t.Error("array created after the panic was freed by a later scope, want the panicked scope to be closed")
	}
}

func TestKeepMovesArrayToOuterScope(t *testing.T) {
	pin(t)
	var kept, dropped *Array
	err := Scoped(func() error {
		if err := Scoped(func() error {
			kept = FromFloat32([]float32{1, 2}, 2)
			dropped = FromFloat32([]float32{3, 4}, 2)
			Keep(kept)
			return nil
		}); err != nil {
			return err
		}
		if !kept.Valid() {
			t.Error("kept array is invalid after the inner scope, want it owned by the outer scope")
		}
		if dropped.Valid() {
			t.Error("array that was not kept is still valid after the inner scope")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if kept.Valid() {
		t.Fatal("kept array is still valid after the outer scope, want it freed there")
	}
}

func TestKeepAtOutermostScopeLeavesOwnershipWithCaller(t *testing.T) {
	pin(t)
	var kept *Array
	err := Scoped(func() error {
		kept = FromFloat32([]float32{1, 2}, 2)
		Keep(kept)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !kept.Valid() {
		t.Fatal("kept array is invalid after the outermost scope, want the caller to own it")
	}
	kept.Free()
	if kept.Valid() {
		t.Fatal("array is still valid after Free")
	}
	kept.Free()
}

func TestArraysCreatedOutsideAnyScopeAreCallerOwned(t *testing.T) {
	pin(t)
	a := FromFloat32([]float32{1, 2}, 2)
	if err := Scoped(func() error { return nil }); err != nil {
		t.Fatal(err)
	}
	if !a.Valid() {
		t.Fatal("array created outside any scope was freed by an unrelated scope")
	}
	a.Free()
	if a.Valid() {
		t.Fatal("array is still valid after Free")
	}
}

func TestKeepOutsideAnyScopeDoesNothing(t *testing.T) {
	pin(t)
	a := FromFloat32([]float32{1, 2}, 2)
	defer a.Free()
	Keep(a)
	if !a.Valid() {
		t.Fatal("Keep outside any scope invalidated a caller-owned array")
	}
}

func TestKeepLeavesArraysOfOuterScopesWithTheirOwner(t *testing.T) {
	pin(t)
	var outer *Array
	err := Scoped(func() error {
		outer = FromFloat32([]float32{1, 2}, 2)
		return Scoped(func() error {
			return Scoped(func() error {
				Keep(outer)
				return nil
			})
		})
	})
	if err != nil {
		t.Fatal(err)
	}
	if outer.Valid() {
		t.Fatal("array is still valid after the scope that created it, want Keep from a deeper scope to leave it there")
	}
}

func TestScopedMemoryReturnsToBaseline(t *testing.T) {
	pin(t)
	data := make([]float32, 256*256)
	ClearCache()
	baseline := ActiveMemory()
	for range 1000 {
		err := Scoped(func() error {
			a := FromFloat32(data, 256, 256)
			b := FromFloat32(data, 256, 256)
			Eval(Add(a, b))
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	ClearCache()
	if got := ActiveMemory(); got > baseline+mib {
		t.Fatalf("ActiveMemory() = %d after 1000 scopes, want within 1 MiB of the baseline %d", got, baseline)
	}
}
