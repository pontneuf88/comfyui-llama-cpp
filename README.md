# ComfyUI llama.cpp

ComfyUI custom nodes for calling a local or remote `llama-server` through the llama.cpp OpenAI-compatible API.

This fork is based on `stavsap/comfyui-ollama`, but it is llama.cpp specific and does not depend on Ollama or the Ollama Python client.

## Requirements

- ComfyUI
- A running llama.cpp server exposing:
  - `GET /v1/models`
  - `POST /v1/chat/completions`

No extra Python package is required by this custom node.

## llama-server example

```bash
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080
```

For vision workflows, start `llama-server` with a multimodal model and its matching projector/options as required by your llama.cpp build and model.

## Nodes

- `llama.cpp Connectivity`
  - Base URL defaults to `http://127.0.0.1:8080`.
  - Use `Reconnect` to load model IDs from `/v1/models`.
  - `request_timeout` controls how long ComfyUI waits for a non-streaming response.
- `llama.cpp Options`
  - Sends enabled sampling options to `/v1/chat/completions`.
  - `num_predict` is sent as `max_tokens`.
- `llama.cpp Generate`
  - One-shot chat-completions wrapper.
  - Optional image input is sent as OpenAI-compatible `image_url` content parts.
  - `keep_context` keeps the conversation in the node instance.
  - `max_tokens` controls the maximum response length.
  - `send_on_change_only` reuses the previous result when all request inputs are unchanged.
- `llama.cpp Chat`
  - Multi-turn chat with JSON history output.
  - Pass `history` into another chat node to continue the same conversation.
  - `max_tokens` controls the maximum response length.
  - `send_on_change_only` reuses the previous result for the same node when all request inputs are unchanged.
- `llama.cpp Save History` and `llama.cpp Load History`
  - Save/load history strings as PNG metadata under `saved_context`.

## JSON output

Selecting `json` adds:

```json
{"response_format": {"type": "json_object"}}
```

The model still needs to be instructed to return valid JSON in the prompt or system prompt.

## Thinking

When `think` is enabled, the request includes:

```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

If the server returns `reasoning_content`, it is exposed through the `thinking` output.

## Timeout notes

The node uses non-streaming chat completions. Long responses therefore return only when generation finishes.

If ComfyUI times out while llama-server is still decoding tokens:

- lower `max_tokens` in `llama.cpp Generate` or `llama.cpp Chat`;
- or lower `num_predict` in `llama.cpp Options` if that option is enabled;
- increase `request_timeout` in `llama.cpp Connectivity`;
- or start `llama-server` with a larger `--timeout`.
