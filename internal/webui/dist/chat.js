// packages/local-chat/src/sse.ts
function createSseParser(onData) {
  let buffer = "";
  let dataLines = [];
  function dispatch() {
    if (dataLines.length > 0)
      onData(dataLines.join(`
`));
    dataLines = [];
  }
  function line(text) {
    if (text === "")
      return dispatch();
    const colon = text.indexOf(":");
    const field = colon === -1 ? text : text.slice(0, colon);
    if (field !== "data")
      return;
    const value = colon === -1 ? "" : text.slice(colon + 1);
    dataLines.push(value.startsWith(" ") ? value.slice(1) : value);
  }
  return {
    push(chunk) {
      buffer += chunk;
      let newline;
      while ((newline = buffer.indexOf(`
`)) !== -1) {
        const end = buffer[newline - 1] === "\r" ? newline - 1 : newline;
        line(buffer.slice(0, end));
        buffer = buffer.slice(newline + 1);
      }
    },
    end() {
      if (buffer !== "")
        line(buffer.endsWith("\r") ? buffer.slice(0, -1) : buffer);
      buffer = "";
      dispatch();
    }
  };
}

// packages/local-chat/src/client.ts
function isLoopbackHost(hostname) {
  const host = hostname.toLowerCase();
  return host === "127.0.0.1" || host === "[::1]" || host === "localhost" || host.endsWith(".localhost");
}
var EXAMPLE_SERVER = "http://127.0.0.1:8080";
function normaliseServerUrl(input, page) {
  const text = input.trim();
  if (text === "")
    throw new Error(`Enter the address of your model server, for example ${EXAMPLE_SERVER}.`);
  let url;
  try {
    url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(text) ? text : `http://${text}`);
  } catch {
    throw new Error(`"${text}" is not a valid address. Use one such as ${EXAMPLE_SERVER}.`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`The address must be an http or https address, for example ${EXAMPLE_SERVER}.`);
  }
  const isOwnServer = !page.hosted && url.origin === page.origin;
  if (!isLoopbackHost(url.hostname) && !isOwnServer) {
    throw new Error(!page.hosted && !isLoopbackHost(page.hostname) ? `Only this computer is allowed, or ${page.origin}, which served this page: ${url.host} is neither.` : `Only this computer is allowed: ${url.hostname} is another machine. Use an address such as ${EXAMPLE_SERVER}.`);
  }
  const path = url.pathname.replace(/\/+$/, "").replace(/\/v1$/, "");
  return `${url.origin}${path}`;
}
async function send(options, path, init) {
  const crossesIntoLoopback = options.pageIsPublic && isLoopbackHost(new URL(options.server).hostname);
  const request = crossesIntoLoopback ? { ...init, targetAddressSpace: "loopback" } : init;
  let response;
  try {
    response = await (options.fetch ?? fetch)(`${options.server}${path}`, request);
  } catch {
    throw connectionError(options, init.signal);
  }
  if (response.status === 403) {
    throw new ChatError("forbidden-origin", `${options.server} does not accept requests from this page.`, 403);
  }
  if (!response.ok) {
    throw new ChatError("http", `${options.server} answered HTTP ${response.status}.`, response.status);
  }
  return response;
}
function connectionError(options, signal) {
  return signal?.aborted ? new ChatError("aborted", "Stopped.") : new ChatError("unreachable", `No answer from ${options.server}.`);
}
function isRecord(value) {
  return typeof value === "object" && value !== null;
}
function parseJson(text, what) {
  try {
    return JSON.parse(text);
  } catch {
    throw new ChatError("protocol", `${what} is not JSON. Is this an OpenAI-compatible server?`);
  }
}
async function listModels(options) {
  const response = await send(options, "/v1/models", {});
  const text = await response.text().catch(() => {
    throw connectionError(options, undefined);
  });
  const body = parseJson(text, "The model list");
  const models = isRecord(body) ? body.data : undefined;
  if (!Array.isArray(models))
    throw new ChatError("protocol", 'The model list has no "data" array.');
  return models.map((model) => {
    if (!isRecord(model) || typeof model.id !== "string")
      throw new ChatError("protocol", 'A model in the list has no "id".');
    return model.id;
  });
}

class ChatError extends Error {
  kind;
  status;
  constructor(kind, message, status) {
    super(message);
    this.name = "ChatError";
    this.kind = kind;
    this.status = status;
  }
}
async function* streamChat(options, request, signal) {
  const response = await send(options, "/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: request.model,
      messages: request.messages,
      max_tokens: request.maxTokens,
      temperature: request.temperature,
      stream: true
    }),
    signal
  });
  if (!response.body)
    throw new ChatError("protocol", "The server answered without a reply stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder;
  const events = [];
  const parser = createSseParser((data) => events.push(data));
  try {
    for (;; ) {
      const { done, value } = await reader.read().catch(() => {
        throw connectionError(options, signal);
      });
      if (done)
        parser.end();
      else
        parser.push(decoder.decode(value, { stream: true }));
      for (const data of events.splice(0)) {
        if (data === "[DONE]")
          return;
        const delta = toDelta(data);
        if (delta)
          yield delta;
      }
      if (done)
        return;
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
}
function errorText(error) {
  if (typeof error === "string")
    return error;
  return isRecord(error) && typeof error.message === "string" ? error.message : JSON.stringify(error);
}
function toDelta(data) {
  const event = parseJson(data, "A stream event");
  if (isRecord(event) && event.error != null)
    throw new ChatError("protocol", `The server reported an error: ${errorText(event.error)}`);
  const choice = isRecord(event) && Array.isArray(event.choices) ? event.choices[0] : undefined;
  if (!isRecord(choice))
    return;
  const delta = {};
  if (isRecord(choice.delta) && typeof choice.delta.content === "string" && choice.delta.content !== "") {
    delta.content = choice.delta.content;
  }
  if (typeof choice.finish_reason === "string")
    delta.finishReason = choice.finish_reason;
  return delta.content === undefined && delta.finishReason === undefined ? undefined : delta;
}

// packages/local-chat/src/permission.ts
async function queryLoopbackPermission(permissions) {
  if (!permissions)
    return "prompt";
  try {
    const { state } = await permissions.query({ name: "loopback-network" });
    return state === "granted" || state === "denied" ? state : "prompt";
  } catch {
    return "prompt";
  }
}

// packages/local-chat/src/state.ts
function initialState(server) {
  return { phase: "disconnected", server, models: [], model: "", messages: [], draft: "" };
}
var PHASE_FOR_PERMISSION = {
  granted: "disconnected",
  prompt: "needs-permission",
  denied: "denied"
};
function withPartialReply(state) {
  return state.draft === "" ? state.messages : [...state.messages, { role: "assistant", content: state.draft }];
}
function stopped(state) {
  return state.phase === "generating" ? { ...state, phase: "connected", messages: withPartialReply(state), draft: "" } : state;
}
var BEFORE_CONNECTING = ["disconnected", "needs-permission", "denied"];
function reduce(state, event) {
  switch (event.type) {
    case "permission":
      return BEFORE_CONNECTING.includes(state.phase) ? { ...state, phase: PHASE_FOR_PERMISSION[event.state] } : state;
    case "connect":
      if (state.phase === "connecting" || state.phase === "generating")
        return state;
      return { ...state, phase: "connecting", server: event.server, error: undefined };
    case "connected": {
      if (state.phase !== "connecting")
        return state;
      const model = event.models.includes(state.model) ? state.model : event.models[0] ?? "";
      return { ...state, phase: "connected", models: event.models, model };
    }
    case "send": {
      const text = event.text.trim();
      if (state.phase !== "connected" || text === "")
        return state;
      return { ...state, phase: "generating", messages: [...state.messages, { role: "user", content: text }] };
    }
    case "delta":
      return state.phase === "generating" ? { ...state, draft: state.draft + event.content } : state;
    case "finished":
      if (state.phase !== "generating")
        return state;
      return { ...state, phase: "connected", messages: [...state.messages, { role: "assistant", content: state.draft }], draft: "" };
    case "stop":
      return stopped(state);
    case "failed":
      if (event.error?.kind === "aborted")
        return stopped(state);
      return { ...state, phase: "error", messages: withPartialReply(state), draft: "", error: event.error };
    case "select-model":
      return state.models.includes(event.model) ? { ...state, model: event.model } : state;
    case "clear":
      return state.phase === "generating" ? state : { ...state, messages: [], draft: "" };
  }
}
function startCommand(origin) {
  return `mlx_lm.server --model <model> --allowed-origins ${origin}`;
}
function fallback(state) {
  return `open ${state.server}/ instead: a server that ships this chat shows it there, and the browser does not get in the way`;
}
function guidance(state, page) {
  const local = URL.canParse(state.server) && isLoopbackHost(new URL(state.server).hostname);
  const start = page.hosted ? `Start a model server on this computer (${startCommand(page.origin)}), then press Connect.` : `Press Connect to reach the model server at ${state.server}.`;
  switch (state.phase) {
    case "disconnected":
      return `${start} Nothing is sent until you press Connect, and then only to ${local ? "your own computer" : "that server"}.`;
    case "needs-permission":
      return `${start} Your browser will ask whether this page may reach apps on this computer: choose Allow.`;
    case "denied":
      return "Your browser is set to stop this page from reaching your computer. Open the site settings for this page " + "(in Chrome: the icon at the left of the address bar, then Site settings), set the permission for apps on this " + `device or the local network to Allow, and reload.${page.hosted ? ` Or ${fallback(state)}.` : ""}`;
    case "connecting":
      return `Connecting to ${state.server}…`;
    case "connected":
      return `Connected to ${state.server}. Your messages go to ${local ? "this computer" : "that server"} and nowhere else.`;
    case "generating":
      return "The model is replying. Press Stop to end the reply early.";
    case "error":
      return errorGuidance(state, page);
  }
}
function errorGuidance(state, page) {
  switch (state.error?.kind) {
    case "unreachable":
      return page.hosted ? `Could not reach ${state.server}. Start a model server on this computer that allows this page: ` + `${startCommand(page.origin)}. Then press Connect. If your browser still blocks the connection, ${fallback(state)}.` : `Could not reach ${state.server}. Check that the model server is still running, then press Connect.`;
    case "forbidden-origin":
      return `${state.server} refused this page. Restart the server so that it allows this origin: ` + `${startCommand(page.origin)}. Then press Connect.`;
    case "http":
      return `${state.error.message} The server's own output says why. Press Connect to try again.`;
    case "protocol":
      return `${state.error.message} This chat needs an OpenAI-compatible server. Press Connect to try again.`;
    case "invalid-url":
      return state.error.message;
    default:
      return "Something went wrong. Press Connect to try again.";
  }
}

// packages/local-chat/src/ui.ts
var ROLE_LABELS = { system: "System", user: "You", assistant: "Model" };
function el(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined)
    node.textContent = text;
  return node;
}
function labelled(text, control) {
  const label = el("label", "lc-field");
  label.append(el("span", "lc-label", text), control);
  return label;
}
function button(type, className, text) {
  const node = el("button", `lc-button ${className}`, text);
  node.type = type;
  return node;
}
function bubble(role, text) {
  const node = el("div", `lc-message lc-${role}`);
  node.append(el("span", "lc-role", ROLE_LABELS[role]), text);
  return node;
}
function describe(error) {
  if (error instanceof ChatError)
    return { kind: error.kind, message: error.message };
  return { kind: "protocol", message: error instanceof Error ? error.message : String(error) };
}
function mount(root, options) {
  const page = { hostname: location.hostname, origin: location.origin, hosted: options.hosted };
  const pageIsPublic = !isLoopbackHost(location.hostname);
  let state = initialState(options.defaultServer);
  let generation;
  const server = el("input", "lc-input");
  server.type = "text";
  server.inputMode = "url";
  server.autocomplete = "off";
  server.spellcheck = false;
  server.value = options.defaultServer;
  const connect = button("submit", "lc-primary", "Connect");
  const model = el("select", "lc-input");
  const connection = el("form", "lc-connect");
  connection.append(labelled("Server address", server), labelled("Model", model), connect);
  const status = el("p", "lc-status");
  status.setAttribute("role", "status");
  const messages = el("div", "lc-messages");
  messages.setAttribute("aria-live", "polite");
  const draftText = el("p", "lc-text");
  const draft = bubble("assistant", draftText);
  draft.classList.add("lc-draft");
  const conversation = el("div", "lc-conversation");
  conversation.tabIndex = 0;
  conversation.setAttribute("role", "region");
  conversation.setAttribute("aria-label", "Conversation");
  conversation.append(messages, draft);
  const input = el("textarea", "lc-input");
  input.rows = 3;
  input.placeholder = "Enter sends, Shift+Enter adds a line";
  const send = button("submit", "lc-primary", "Send");
  const stop = button("button", "lc-secondary", "Stop");
  const clear = button("button", "lc-secondary", "Clear");
  const actions = el("div", "lc-actions");
  actions.append(send, stop, clear);
  const compose = el("form", "lc-compose");
  compose.append(labelled("Message", input), actions);
  const chat = el("div", "lc");
  chat.append(connection, status, conversation, compose);
  root.replaceChildren(chat);
  function render(previous) {
    const busy = state.phase === "connecting" || state.phase === "generating";
    chat.dataset.phase = state.phase;
    status.textContent = guidance(state, page);
    server.disabled = busy;
    connect.disabled = busy;
    send.disabled = state.phase !== "connected";
    stop.disabled = state.phase !== "generating";
    clear.disabled = state.phase === "generating" || state.messages.length === 0;
    if (state.models !== previous?.models) {
      model.replaceChildren(...state.models.map((id) => {
        const option = el("option", "", id);
        option.value = id;
        return option;
      }));
    }
    model.value = state.model;
    model.disabled = state.phase !== "connected" || state.models.length === 0;
    if (state.messages.length < messages.childElementCount)
      messages.replaceChildren();
    for (const message of state.messages.slice(messages.childElementCount)) {
      messages.append(bubble(message.role, el("p", "lc-text", message.content)));
    }
    draftText.textContent = state.draft;
    draft.hidden = state.phase !== "generating";
    if (state.messages !== previous?.messages || state.draft !== previous.draft) {
      conversation.scrollTop = conversation.scrollHeight;
    }
  }
  function dispatch(event) {
    const previous = state;
    state = reduce(state, event);
    render(previous);
  }
  async function connectToServer() {
    let address;
    try {
      address = normaliseServerUrl(server.value, page);
    } catch (error) {
      dispatch({ type: "failed", error: { kind: "invalid-url", message: error instanceof Error ? error.message : String(error) } });
      return;
    }
    server.value = address;
    dispatch({ type: "connect", server: address });
    try {
      dispatch({ type: "connected", models: await listModels({ server: address, pageIsPublic }) });
    } catch (error) {
      dispatch({ type: "failed", error: describe(error) });
    }
  }
  async function sendMessage() {
    const before = state;
    dispatch({ type: "send", text: input.value });
    if (state === before)
      return;
    input.value = "";
    generation = new AbortController;
    try {
      const request = { model: state.model || undefined, messages: state.messages };
      for await (const delta of streamChat({ server: state.server, pageIsPublic }, request, generation.signal)) {
        if (delta.content !== undefined)
          dispatch({ type: "delta", content: delta.content });
      }
      dispatch({ type: "finished" });
    } catch (error) {
      dispatch({ type: "failed", error: describe(error) });
    }
  }
  connection.addEventListener("submit", (event) => {
    event.preventDefault();
    connectToServer();
  });
  compose.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing)
      return;
    event.preventDefault();
    sendMessage();
  });
  stop.addEventListener("click", () => {
    generation?.abort();
    dispatch({ type: "stop" });
  });
  clear.addEventListener("click", () => dispatch({ type: "clear" }));
  model.addEventListener("change", () => dispatch({ type: "select-model", model: model.value }));
  render(undefined);
  if (options.hosted && pageIsPublic) {
    queryLoopbackPermission(navigator.permissions).then((permission) => dispatch({ type: "permission", state: permission }));
  }
}

// packages/local-chat/src/standalone.ts
var root = document.getElementById("local-chat");
if (root)
  mount(root, { defaultServer: location.origin, hosted: false });
