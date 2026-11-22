package tokenizer

import (
	"cmp"
	"slices"
	"strings"
)

// addedToken is one entry of tokenizer.json "added_tokens": text that maps
// straight to an id, bypassing normalisation, splitting and BPE.
type addedToken struct {
	content string
	id      int32
}

// addedTokens finds added tokens in raw text.
type addedTokens struct {
	// byFirstByte[b] holds the tokens whose content starts with byte b,
	// longest first.
	byFirstByte [256][]addedToken
}

// newAddedTokens indexes the added tokens of tokenizer.json, none of which
// may have empty content.
func newAddedTokens(tokens []addedTokenFile) *addedTokens {
	index := &addedTokens{}
	for _, token := range tokens {
		first := token.Content[0]
		index.byFirstByte[first] = append(index.byFirstByte[first], addedToken{token.Content, token.ID})
	}
	for _, group := range index.byFirstByte {
		slices.SortStableFunc(group, func(a, b addedToken) int {
			return cmp.Compare(len(b.content), len(a.content))
		})
	}
	return index
}

// find locates the first added token in text: text[start:end] is the token
// and id is its id. Of several tokens starting at the same place the longest
// wins, which is the reference's leftmost-longest matching. ok is false when
// text holds no added token.
func (a *addedTokens) find(text string) (start, end int, id int32, ok bool) {
	for i := range len(text) {
		for _, token := range a.byFirstByte[text[i]] {
			if strings.HasPrefix(text[i:], token.content) {
				return i, i + len(token.content), token.id, true
			}
		}
	}
	return 0, 0, 0, false
}
