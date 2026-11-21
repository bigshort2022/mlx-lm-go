package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	"github.com/bigshort2022/mlx-lm-go/internal/engine"
)

// defaultContextSize is the reference's default for the repetition, presence
// and frequency penalty windows.
const defaultContextSize = 20

// object is a decoded JSON request body.
type object map[string]json.RawMessage

// decodeBody parses the body of a completion request. Its error messages are
// the reference's.
func decodeBody(raw []byte) (object, error) {
	var body object
	err := json.Unmarshal(raw, &body)
	var wrongType *json.UnmarshalTypeError
	switch {
	case errors.As(err, &wrongType), err == nil && body == nil:
		return nil, errors.New("Request should be a JSON dictionary")
	case err != nil:
		return nil, fmt.Errorf("Invalid JSON in request body: %v", err)
	}
	return body, nil
}

// reader reads typed fields out of a request body. It remembers the first
// problem it meets, so a caller can read a group of fields and check err once.
// Messages use the reference's wording.
type reader struct {
	body object
	err  error
}

// raw returns the JSON of a field. An absent or null field counts as not given.
func (r *reader) raw(name string) (json.RawMessage, bool) {
	v, ok := r.body[name]
	return v, ok && string(v) != "null"
}

func (r *reader) fail(format string, args ...any) {
	if r.err == nil {
		r.err = fmt.Errorf(format, args...)
	}
}

// boolean returns the named boolean field, or def when it is not given.
func (r *reader) boolean(name string, def bool) bool {
	v, ok := r.raw(name)
	if !ok {
		return def
	}
	switch string(v) {
	case "true":
		return true
	case "false":
		return false
	}
	r.fail("%s must be of type bool", name)
	return def
}

// limit is a bound on a numeric field.
type limit struct {
	lower bool // the bound is a minimum; otherwise a maximum
	bound float64
}

func atLeast(bound float64) limit { return limit{lower: true, bound: bound} }
func atMost(bound float64) limit  { return limit{bound: bound} }

// check records a failure if v breaks one of the limits.
func (r *reader) check(name string, v float64, limits []limit) {
	for _, l := range limits {
		switch {
		case l.lower && v < l.bound:
			r.fail("%s must be at least %v", name, l.bound)
		case !l.lower && v > l.bound:
			r.fail("%s must be at most %v", name, l.bound)
		}
	}
}

// integer returns the named integer field, or def when it is not given.
func (r *reader) integer(name string, def int, limits ...limit) int {
	v, ok := r.raw(name)
	if !ok {
		return def
	}
	n, err := strconv.Atoi(string(v))
	if err != nil {
		r.fail("%s must be of type int", name)
		return def
	}
	r.check(name, float64(n), limits)
	return n
}

// number returns the named numeric field, or def when it is not given.
func (r *reader) number(name string, def float32, limits ...limit) float32 {
	v, ok := r.raw(name)
	if !ok {
		return def
	}
	f, err := strconv.ParseFloat(string(v), 64)
	if err != nil {
		r.fail("%s must be of type ('float', 'int')", name)
		return def
	}
	r.check(name, f, limits)
	return float32(f)
}

// dict returns the named field if it is a JSON object, or nil when it is not
// given.
func (r *reader) dict(name string) json.RawMessage {
	v, ok := r.raw(name)
	if !ok {
		return nil
	}
	if !isJSONObject(v) {
		r.fail("%s must be of type dict", name)
		return nil
	}
	return v
}

// text returns the named string field, or def when it is not given.
func (r *reader) text(name, def string) string {
	v, ok := r.raw(name)
	if !ok {
		return def
	}
	var s string
	if err := json.Unmarshal(v, &s); err != nil {
		r.fail("%s must be of type str", name)
		return def
	}
	return s
}

// stop returns the stop strings. The field may hold one string or a list.
func (r *reader) stop() []string {
	v, ok := r.raw("stop")
	if !ok {
		return nil
	}
	var one string
	if json.Unmarshal(v, &one) == nil {
		return []string{one}
	}
	var many []string
	if err := json.Unmarshal(v, &many); err != nil {
		r.fail("stop must be a string or a list of strings")
		return nil
	}
	return many
}

// includeUsage reports whether stream_options asks for a usage event at the
// end of the stream.
func (r *reader) includeUsage() bool {
	v := r.dict("stream_options")
	if v == nil {
		return false
	}
	var options object
	_ = json.Unmarshal(v, &options) // dict has checked that v is an object
	inner := reader{body: options}
	include := inner.boolean("include_usage", false)
	if inner.err != nil {
		r.fail("%v", inner.err)
	}
	return include
}

// logitBias returns logit_bias as a map from token id to bias, or nil when it
// is not given.
func (r *reader) logitBias() map[int32]float32 {
	v := r.dict("logit_bias")
	if v == nil {
		return nil
	}
	var entries map[string]float32
	if err := json.Unmarshal(v, &entries); err != nil {
		r.fail("logit_bias must be a dict of int to float")
		return nil
	}
	bias := make(map[int32]float32, len(entries))
	for key, value := range entries {
		id, err := strconv.ParseInt(key, 10, 32)
		if err != nil {
			r.fail("logit_bias must be a dict of int to float")
			return nil
		}
		bias[int32(id)] = value
	}
	return bias
}

// seed returns the sampling seed, or nil when it is not given.
func (r *reader) seed() *uint64 {
	if _, ok := r.raw("seed"); !ok {
		return nil
	}
	seed := uint64(r.integer("seed", 0, atLeast(0)))
	return &seed
}

// unsupportedFields are request fields of the reference that this server does
// not implement yet.
var unsupportedFields = []string{
	"adapters", "draft_model", "num_draft_tokens", "role_mapping", "tools",
	"logprobs", "top_logprobs", "xtc_probability", "xtc_threshold",
}

// rejectUnsupported fails for an unsupported field that asks for something.
// A value that leaves the feature off is accepted, so that clients which
// always send, say, "logprobs": false keep working.
func (r *reader) rejectUnsupported() {
	for _, name := range unsupportedFields {
		if v, ok := r.raw(name); ok && !isOff(v) {
			r.fail("%s is not supported yet", name)
		}
	}
}

// isOff reports whether a JSON value is one that Python treats as false:
// null, false, 0, "", [] or {}.
func isOff(raw json.RawMessage) bool {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return false
	}
	switch v := value.(type) {
	case bool:
		return !v
	case float64:
		return v == 0
	case string:
		return v == ""
	case []any:
		return len(v) == 0
	case map[string]any:
		return len(v) == 0
	}
	return value == nil
}

// defaultModel is the reference's name for the model loaded at startup.
const defaultModel = "default_model"

// model returns the requested model name, which is echoed in the response.
// This server cannot switch models, so it accepts only the default name and
// the model it serves.
func (s *server) model(r *reader) string {
	name := r.text("model", defaultModel)
	if name != defaultModel && name != s.cfg.Model {
		r.fail("model %q is not available; this server only serves %q", name, s.cfg.Model)
	}
	return name
}

// maxTokens returns the generation budget. Like the reference it prefers
// max_completion_tokens to max_tokens and falls back to --max-tokens.
func (s *server) maxTokens(r *reader) int {
	if _, ok := r.raw("max_completion_tokens"); ok {
		return r.integer("max_completion_tokens", 0, atLeast(0))
	}
	return r.integer("max_tokens", s.cfg.MaxTokens, atLeast(0))
}

// sampling reads the sampling fields. Fields the request leaves out take the
// server's defaults.
func (s *server) sampling(r *reader) engine.Sampling {
	return engine.Sampling{
		Temperature:           r.number("temperature", s.cfg.Temperature, atLeast(0)),
		TopP:                  r.number("top_p", s.cfg.TopP, atLeast(0), atMost(1)),
		TopK:                  r.integer("top_k", s.cfg.TopK, atLeast(0)),
		MinP:                  r.number("min_p", s.cfg.MinP, atLeast(0), atMost(1)),
		RepetitionPenalty:     r.number("repetition_penalty", 0, atLeast(0)),
		RepetitionContextSize: r.integer("repetition_context_size", defaultContextSize, atLeast(0)),
		PresencePenalty:       r.number("presence_penalty", 0),
		PresenceContextSize:   r.integer("presence_context_size", defaultContextSize, atLeast(0)),
		FrequencyPenalty:      r.number("frequency_penalty", 0),
		FrequencyContextSize:  r.integer("frequency_context_size", defaultContextSize, atLeast(0)),
		LogitBias:             r.logitBias(),
		Seed:                  r.seed(),
	}
}
