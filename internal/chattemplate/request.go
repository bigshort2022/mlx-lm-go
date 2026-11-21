package chattemplate

import (
	"errors"
	"fmt"
	"maps"
)

// reservedKwargs are the parameters of Python's apply_chat_template besides
// documents. Python never passes one on to the template: it refuses the ones
// the caller already gave as duplicates, and acts on the others itself, for
// instance by rendering a different template. None is a template variable,
// so none may arrive among the kwargs.
var reservedKwargs = []string{
	"messages", "tools", "add_generation_prompt",
	"self", "conversation", "chat_template", "continue_final_message",
	"tokenize", "padding", "truncation", "max_length",
	"return_tensors", "return_dict", "return_assistant_tokens_mask", "tokenizer_kwargs",
}

// variables builds the template's top-level variables the way
// apply_chat_template does: special tokens, overridden by kwargs, plus the
// request fields. It rejects what apply_chat_template rejects before it
// renders.
func (t *Template) variables(in Input) (map[string]any, error) {
	variables := map[string]any{"documents": nil}
	for name, token := range t.specialTokens {
		variables[name] = token
	}
	kwargs, err := decodeOptional(in.Kwargs)
	if err != nil {
		return nil, fmt.Errorf("kwargs: %w", err)
	}
	switch kwargs := kwargs.(type) {
	case nil:
	case *dict:
		for _, name := range reservedKwargs {
			if _, present := kwargs.values[name]; present {
				return nil, fmt.Errorf("kwargs: %q is an argument of apply_chat_template, not a template variable", name)
			}
		}
		if err := checkObjects(kwargs.values["documents"], "document"); err != nil {
			return nil, fmt.Errorf("kwargs: documents: %w", err)
		}
		maps.Copy(variables, kwargs.values)
	default:
		return nil, errors.New("kwargs: not a JSON object")
	}

	messages, err := decodeValue(in.Messages)
	if err != nil {
		return nil, fmt.Errorf("messages: %w", err)
	}
	conversation, ok := messages.([]any)
	switch {
	case !ok:
		return nil, errors.New("messages: not a JSON array")
	case len(conversation) == 0:
		return nil, errors.New("messages: the conversation is empty")
	}
	if _, nested := conversation[0].([]any); nested {
		// Python takes a list whose first item is a list for several
		// conversations and returns one prompt for each.
		return nil, errors.New("messages: a batch of conversations is not supported")
	}
	tools, err := decodeOptional(in.Tools)
	if err != nil {
		return nil, fmt.Errorf("tools: %w", err)
	}
	if err := checkObjects(tools, "tool"); err != nil {
		return nil, fmt.Errorf("tools: %w", err)
	}
	variables["messages"] = messages
	variables["tools"] = tools
	variables["add_generation_prompt"] = in.AddGenerationPrompt
	return variables, nil
}

// checkObjects requires value to be none or a list of objects, which is what
// apply_chat_template requires of tools and of documents.
func checkObjects(value any, item string) error {
	if value == nil {
		return nil
	}
	list, ok := value.([]any)
	if !ok {
		return errors.New("not a JSON array")
	}
	for i, element := range list {
		if _, ok := element.(*dict); !ok {
			return fmt.Errorf("%s %d is not a JSON object", item, i)
		}
	}
	return nil
}
