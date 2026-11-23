// lbprobe fires N concurrent chat-completion requests at an endpoint and
// reports how traffic spread across replicas (x-served-by), status codes,
// and latency percentiles. It exits 1 if spread skew or error rate exceed
// limits, so it can gate a CI stage or a canary.
//
//	go run . -url http://localhost:8080/v1/chat/completions -n 300 -c 20 -host inference.local
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"
)

type sample struct {
	replica string
	status  int
	latency time.Duration
	err     error
}

type Report struct {
	Requests  int            `json:"requests"`
	Errors    int            `json:"errors"`
	ErrorRate float64        `json:"error_rate"`
	Status    map[int]int    `json:"status"`
	Replicas  map[string]int `json:"replicas"`
	Skew      float64        `json:"skew"` // (max-min)/mean across replicas
	P50ms     float64        `json:"p50_ms"`
	P99ms     float64        `json:"p99_ms"`
}

// Percentile uses nearest-rank so the result is always an observed value.
func Percentile(sorted []time.Duration, pct float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	rank := int(math.Ceil(pct / 100 * float64(len(sorted))))
	if rank < 1 {
		rank = 1
	}
	return sorted[rank-1]
}

// Skew is (max-min)/mean of per-replica counts; 0 is a perfect spread.
func Skew(counts map[string]int) float64 {
	if len(counts) == 0 {
		return 0
	}
	lo, hi, total := -1, 0, 0
	for _, c := range counts {
		if lo < 0 || c < lo {
			lo = c
		}
		if c > hi {
			hi = c
		}
		total += c
	}
	mean := float64(total) / float64(len(counts))
	return float64(hi-lo) / mean
}

func Summarize(samples []sample) Report {
	r := Report{Requests: len(samples), Status: map[int]int{}, Replicas: map[string]int{}}
	var lat []time.Duration
	for _, s := range samples {
		if s.err != nil || s.status >= 400 {
			r.Errors++
		}
		if s.err == nil {
			r.Status[s.status]++
			lat = append(lat, s.latency)
		}
		if s.err == nil && s.status < 400 {
			r.Replicas[s.replica]++
		}
	}
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	if r.Requests > 0 {
		r.ErrorRate = float64(r.Errors) / float64(r.Requests)
	}
	r.Skew = Skew(r.Replicas)
	r.P50ms = float64(Percentile(lat, 50).Microseconds()) / 1000
	r.P99ms = float64(Percentile(lat, 99).Microseconds()) / 1000
	return r
}

func fire(client *http.Client, url, host, model string) sample {
	body, _ := json.Marshal(map[string]interface{}{
		"model":    model,
		"messages": []map[string]string{{"role": "user", "content": "ping"}},
	})
	req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	req.Header.Set("content-type", "application/json")
	if host != "" {
		req.Host = host
	}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return sample{err: err}
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return sample{replica: resp.Header.Get("x-served-by"), status: resp.StatusCode, latency: time.Since(start)}
}

func main() {
	url := flag.String("url", "http://127.0.0.1:8080/v1/chat/completions", "chat completions URL")
	host := flag.String("host", "", "Host header override (ingress virtual host)")
	model := flag.String("model", "llama3.1-8b", "model id")
	n := flag.Int("n", 200, "total requests")
	c := flag.Int("c", 16, "concurrency")
	maxSkew := flag.Float64("max-skew", 0.5, "fail if replica skew exceeds this")
	maxErr := flag.Float64("max-error-rate", 0.01, "fail if error rate exceeds this")
	flag.Parse()

	client := &http.Client{Timeout: 30 * time.Second}
	jobs := make(chan struct{})
	out := make(chan sample, *n)
	var wg sync.WaitGroup
	for i := 0; i < *c; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range jobs {
				out <- fire(client, *url, *host, *model)
			}
		}()
	}
	for i := 0; i < *n; i++ {
		jobs <- struct{}{}
	}
	close(jobs)
	wg.Wait()
	close(out)

	var samples []sample
	for s := range out {
		samples = append(samples, s)
	}
	r := Summarize(samples)
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(r)

	if r.Skew > *maxSkew || r.ErrorRate > *maxErr {
		fmt.Fprintf(os.Stderr, "FAIL skew=%.2f (max %.2f) error_rate=%.4f (max %.4f)\n", r.Skew, *maxSkew, r.ErrorRate, *maxErr)
		os.Exit(1)
	}
}
