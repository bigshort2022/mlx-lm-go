// Package chattemplate renders a model's Jinja chat template the way
// tokenizer.apply_chat_template does in Hugging Face transformers.
//
// # What is promised
//
// For a template built from the constructs listed below, Render returns
// exactly the string Python returns for the same request, and an error
// wherever Python raises. Everything else is an error too: a construct this
// package does not implement stops Parse with an error that names it, and an
// operation it does not implement stops Render. It never approximates.
//
// The reference is transformers 5.7.0 with jinja2 3.1.6 on Python 3.13. The
// fixtures under testdata record what the reference does;
// tools/oracle/make_template_differential.py writes them.
//
// # Supported constructs
//
//   - Statements: if, elif, else; for, with unpacking, optional filtering and
//     an else branch. Loop metadata (index, index0, revindex, revindex0,
//     first, last, length, depth, depth0, previtem, nextitem) counts only
//     accepted items. Set of a name or of a namespace attribute; macro,
//     with positional, keyword and default arguments. Comments and whitespace
//     control, with trim_blocks and lstrip_blocks on as Hugging Face sets
//     them.
//   - Expressions: string and integer literals, parenthesized tuples, true,
//     false and none; names; attribute, item and slice access; calls; + and -,
//     ~, the comparisons, in and not in, and, or, not, and "a if b else c".
//   - Filters: default (with optional fallback and boolean mode), items, length,
//     safe, string, trim, and Hugging Face's tojson with indent and ensure_ascii.
//   - Tests: defined, undefined, none, true, false, string, mapping,
//     iterable, sequence.
//   - String methods: startswith, endswith, split with a separator, strip,
//     lstrip, rstrip.
//   - Functions: namespace and raise_exception.
//
// # Deliberate limits
//
// Python accepts each of the following; this package returns an error.
//
// Requests:
//
//   - Messages must be one conversation: a JSON array that does not start
//     with an array. Python takes a list of lists for a batch and returns a
//     prompt for each conversation, and hands anything that is not a list
//     to the template as it is.
//   - Kwargs may not name a parameter of apply_chat_template: chat_template,
//     continue_final_message, tokenize, padding, truncation, max_length,
//     return_tensors, return_dict, return_assistant_tokens_mask,
//     tokenizer_kwargs, conversation, self, or the request fields messages,
//     tools and add_generation_prompt. Python rejects some as duplicates and
//     acts on the others, for instance rendering the template it is handed
//     in chat_template. The documents parameter is allowed and must be a
//     list of objects or null; Python also lets an empty object through.
//   - JSON text may not hold an escape for half a surrogate pair, such as
//     "\ud800", which a Python string can carry and a Go string cannot, nor
//     the NaN and Infinity literals json.loads adds to JSON.
//
// Values:
//
//   - An integer beyond 64 bits can be read from a request, rendered,
//     written by tojson, compared for equality and used as a slice bound,
//     but not added, subtracted, negated or ordered. A sum or difference
//     that leaves the 64-bit range is an error, as is an integer literal
//     beyond it in a template.
//   - Reading one of dict's method names as an attribute, as in d.items or
//     d.get, is an error, as is d["items"] when the dict has no such key;
//     Python returns the method. An attribute Python defines on a str, int,
//     float, bool, list or tuple, other than the string methods above, is an
//     error; so is any attribute of a macro.
//   - A safe string can be written, joined with ~, tested, compared, and
//     passed to string, trim, length, tojson and safe. Any other use is an
//     error, because Python would escape the other operand.
//   - Writing a namespace, a macro, a function, the loop variable or the
//     result of the items filter as text is an error; Python writes a memory
//     address. Ordering two lists is an error.
//   - The string methods take only the arguments listed above, namespace
//     takes only keyword arguments, and tojson takes an indent of at most
//     100 and no positional, separators or sort_keys argument. Apart from
//     default, the other filters take no arguments.
//
// Templates:
//
//   - A macro must be defined at the top level or inside an if, not inside a
//     for body or another macro, and cannot use varargs, kwargs or caller.
//   - At most 150 macro calls can be in progress at once. Python's limit
//     depends on its own call stack and was a little under 200 when measured.
//   - A comment left open at the end of a template is an error; Python drops
//     it.
//   - Statements, operators, literals, filters, tests, functions and loop
//     attributes not listed above are Parse errors.
//
// # Known divergences
//
// Two differences from Python are not errors.
//
// When a list or a mapping is written as text, Python's repr escapes the
// characters that are not printable according to its Unicode tables, version
// 15.1 in Python 3.13; Go decides by its own, newer tables. A character
// added to Unicode since then is therefore escaped there and written as it
// is here. Strings written directly and the tojson filter are not affected,
// and a caller that flattens text content parts before rendering never
// writes such a repr into a prompt.
//
// Load gives a template the special tokens bos_token, eos_token, pad_token
// and unk_token. Python also passes sep_token, cls_token and mask_token when
// the tokenizer has them; a template reading one of those finds it undefined
// here.
package chattemplate
