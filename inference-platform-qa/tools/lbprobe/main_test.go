package main

import (
	"errors"
	"testing"
	"time"
)

func ms(v ...int) []time.Duration {
	out := make([]time.Duration, len(v))
	for i, x := range v {
		out[i] = time.Duration(x) * time.Millisecond
	}
	return out
}

func TestPercentileNearestRank(t *testing.T) {
	d := ms(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	cases := map[float64]time.Duration{50: 5 * time.Millisecond, 99: 10 * time.Millisecond, 1: time.Millisecond}
	for pct, want := range cases {
		if got := Percentile(d, pct); got != want {
			t.Errorf("p%v = %v, want %v", pct, got, want)
		}
	}
	if Percentile(nil, 99) != 0 {
		t.Error("empty input should be 0")
	}
}

func TestSkew(t *testing.T) {
	if s := Skew(map[string]int{"a": 10, "b": 10, "c": 10}); s != 0 {
		t.Errorf("even spread skew = %v", s)
	}
	if s := Skew(map[string]int{"a": 30, "b": 0, "c": 0}); s != 3 {
		t.Errorf("all-on-one skew = %v, want 3", s)
	}
}

func TestSummarizeCountsErrorsAndExcludesThemFromSpread(t *testing.T) {
	r := Summarize([]sample{
		{replica: "a", status: 200, latency: time.Millisecond},
		{replica: "b", status: 200, latency: 2 * time.Millisecond},
		{replica: "a", status: 503, latency: time.Millisecond},
		{err: errors.New("conn refused")},
	})
	if r.Errors != 2 || r.ErrorRate != 0.5 {
		t.Fatalf("errors=%d rate=%v", r.Errors, r.ErrorRate)
	}
	if r.Replicas["a"] != 1 || r.Replicas["b"] != 1 {
		t.Fatalf("replicas=%v", r.Replicas)
	}
	if r.Status[503] != 1 {
		t.Fatalf("status=%v", r.Status)
	}
}
