from __future__ import annotations

import base64
import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from io import BytesIO
from pprint import pprint
from typing import TYPE_CHECKING, Any

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

if TYPE_CHECKING:
    import torch


CATEGORY = "llama.cpp"
DEFAULT_URL = "http://127.0.0.1:8080"
HISTORY_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "saved_context")


@dataclass
class ChatSession:
    messages: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""


CHAT_SESSIONS: dict[str, ChatSession] = {}


def _normalise_url(url: str) -> str:
    return url.rstrip("/")


def _json_request(url: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    target = f"{_normalise_url(url)}{path}"
    data = None
    method = "GET"
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(target, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama.cpp server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not connect to llama.cpp server at {target}: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"llama.cpp server returned invalid JSON from {target}: {body[:500]}") from exc


def _list_models(url: str) -> list[str]:
    response = _json_request(url, "/v1/models")
    models = response.get("data", [])
    names = [model.get("id") for model in models if isinstance(model, dict) and model.get("id")]
    return names


def _filter_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if not options:
        return {}

    mappings = {
        "enable_mirostat": "mirostat",
        "enable_mirostat_eta": "mirostat_eta",
        "enable_mirostat_tau": "mirostat_tau",
        "enable_repeat_penalty": "repeat_penalty",
        "enable_temperature": "temperature",
        "enable_seed": "seed",
        "enable_stop": "stop",
        "enable_num_predict": "max_tokens",
        "enable_top_k": "top_k",
        "enable_top_p": "top_p",
        "enable_min_p": "min_p",
    }

    request_options: dict[str, Any] = {}
    for enabler, request_key in mappings.items():
        if options.get(enabler):
            value_key = enabler.replace("enable_", "")
            value = options.get(value_key)
            if request_key == "stop" and isinstance(value, str):
                value = [item.strip() for item in value.splitlines() if item.strip()] or value
            request_options[request_key] = value

    return request_options


def _images_to_content_parts(prompt: str, images: list[torch.Tensor] | None) -> str | list[dict[str, Any]]:
    if images is None:
        return prompt

    import numpy as np
    from PIL import Image

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        array = 255.0 * image.cpu().numpy()
        pil_image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )
    return content


def _parse_history(history: str | None) -> list[dict[str, Any]]:
    if not history:
        return []

    try:
        parsed = json.loads(history)
    except json.JSONDecodeError as exc:
        raise ValueError("history must be a JSON string produced by LlamaCpp Chat.") from exc

    if isinstance(parsed, dict):
        parsed = parsed.get("messages", [])
    if not isinstance(parsed, list):
        raise ValueError("history JSON must contain a list of messages.")

    return [message for message in parsed if isinstance(message, dict) and "role" in message]


def _history_to_string(messages: list[dict[str, Any]]) -> str:
    history_messages = []
    for message in messages:
        copied = dict(message)
        if isinstance(copied.get("content"), list):
            copied["content"] = _text_from_content(copied["content"])
        history_messages.append(copied)
    return json.dumps({"messages": history_messages}, ensure_ascii=False)


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _build_request(
    model: str,
    messages: list[dict[str, Any]],
    think: bool,
    output_format: str,
    options: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    payload.update(_filter_options(options))

    if output_format == "json":
        payload["response_format"] = {"type": "json_object"}

    if think:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    return payload


def _run_chat(url: str, payload: dict[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    response = _json_request(url, "/v1/chat/completions", payload)
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError(f"llama.cpp response did not contain choices: {response}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        result = _text_from_content(content)
    elif content is None:
        result = ""
    else:
        result = str(content)

    thinking = message.get("reasoning_content")
    if thinking is not None:
        thinking = str(thinking)

    return result, thinking, response


if PromptServer is not None:
    from aiohttp import web

    @PromptServer.instance.routes.post("/llamacpp/get_models")
    async def get_models_endpoint(request):
        data = await request.json()
        url = data.get("url", DEFAULT_URL)
        try:
            return web.json_response(_list_models(url))
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)


class LlamaCppSaveHistory:
    def __init__(self):
        self._base_dir = HISTORY_DIR
        os.makedirs(self._base_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "history": ("LLAMACPP_HISTORY", {"forceInput": True}),
                "filename": ("STRING", {"default": "history"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "llamacpp_save_history"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def llamacpp_save_history(self, filename: str, history: str | None = None):
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        safe_filename = os.path.basename(filename)
        path = os.path.join(self._base_dir, safe_filename)
        metadata = PngInfo()
        metadata.add_text("history", history or "")
        image = Image.new("RGB", (100, 100), (255, 255, 255))
        image.save(path + ".png", pnginfo=metadata)
        return {"ui": {"history": history}}


class LlamaCppLoadHistory:
    def __init__(self):
        self._base_dir = HISTORY_DIR
        os.makedirs(self._base_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        os.makedirs(HISTORY_DIR, exist_ok=True)
        files = [
            filename
            for filename in os.listdir(HISTORY_DIR)
            if os.path.isfile(os.path.join(HISTORY_DIR, filename)) and filename != ".keep"
        ]
        return {"required": {"history_file": (files or [""], {})}}

    CATEGORY = CATEGORY
    RETURN_NAMES = ("history",)
    RETURN_TYPES = ("LLAMACPP_HISTORY",)
    FUNCTION = "llamacpp_load_history"

    def llamacpp_load_history(self, history_file: str):
        from PIL import Image

        if not history_file:
            return ("",)
        with Image.open(os.path.join(self._base_dir, history_file)) as image:
            return (image.info.get("history", ""),)


class LlamaCppOptions:
    @classmethod
    def INPUT_TYPES(cls):
        seed = random.randint(1, 2**31)
        return {
            "required": {
                "enable_mirostat": ("BOOLEAN", {"default": False}),
                "mirostat": ("INT", {"default": 0, "min": 0, "max": 2, "step": 1}),
                "enable_mirostat_eta": ("BOOLEAN", {"default": False}),
                "mirostat_eta": ("FLOAT", {"default": 0.1, "min": 0, "step": 0.1}),
                "enable_mirostat_tau": ("BOOLEAN", {"default": False}),
                "mirostat_tau": ("FLOAT", {"default": 5.0, "min": 0, "step": 0.1}),
                "enable_repeat_penalty": ("BOOLEAN", {"default": False}),
                "repeat_penalty": ("FLOAT", {"default": 1.1, "min": 0, "max": 2, "step": 0.05}),
                "enable_temperature": ("BOOLEAN", {"default": False}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 2, "step": 0.05}),
                "enable_seed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": seed, "min": 0, "max": 2**31, "step": 1}),
                "enable_stop": ("BOOLEAN", {"default": False}),
                "stop": ("STRING", {"default": "", "multiline": True}),
                "enable_num_predict": ("BOOLEAN", {"default": False}),
                "num_predict": ("INT", {"default": 512, "min": 1, "max": 32768, "step": 1}),
                "enable_top_k": ("BOOLEAN", {"default": False}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 100, "step": 1}),
                "enable_top_p": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0, "max": 1, "step": 0.05}),
                "enable_min_p": ("BOOLEAN", {"default": False}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LLAMACPP_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "llamacpp_options"
    CATEGORY = CATEGORY
    DESCRIPTION = "Advanced llama.cpp server generation settings for the OpenAI-compatible chat endpoint."

    def llamacpp_options(self, **kwargs):
        if kwargs.get("debug"):
            print("--- llama.cpp options dump")
            pprint(kwargs)
            print("---------------------------------------------------------")
        return (kwargs,)


class LlamaCppConnectivity:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": DEFAULT_URL,
                        "tooltip": "The base URL of the llama.cpp server.",
                    },
                ),
                "model": (
                    (),
                    {
                        "tooltip": "Select a model exposed by /v1/models. Use Reconnect after starting llama-server.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LLAMACPP_CONNECTIVITY",)
    RETURN_NAMES = ("connection",)
    FUNCTION = "llamacpp_connectivity"
    CATEGORY = CATEGORY
    DESCRIPTION = "Connection settings for a llama.cpp server OpenAI-compatible API."

    def llamacpp_connectivity(self, url: str, model: str):
        return ({"url": url, "model": model},)


class LlamaCppGenerate:
    def __init__(self):
        self.saved_messages: list[dict[str, Any]] | None = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": ("STRING", {"multiline": True, "default": "You are an AI artist."}),
                "prompt": ("STRING", {"multiline": True, "default": "What is art?"}),
                "think": ("BOOLEAN", {"default": False}),
                "keep_context": ("BOOLEAN", {"default": False}),
                "format": (["text", "json"],),
            },
            "optional": {
                "connectivity": ("LLAMACPP_CONNECTIVITY", {"forceInput": False}),
                "options": ("LLAMACPP_OPTIONS", {"forceInput": False}),
                "images": ("IMAGE", {"forceInput": False}),
                "context": ("LLAMACPP_HISTORY", {"forceInput": False}),
                "meta": ("LLAMACPP_META", {"forceInput": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "LLAMACPP_HISTORY", "LLAMACPP_META")
    RETURN_NAMES = ("result", "thinking", "context", "meta")
    FUNCTION = "llamacpp_generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "One-shot text or vision generation through llama.cpp /v1/chat/completions."

    def llamacpp_generate(
        self,
        system: str,
        prompt: str,
        think: bool,
        keep_context: bool,
        format: str,
        context: str | None = None,
        options: dict[str, Any] | None = None,
        connectivity: dict[str, Any] | None = None,
        images: list[torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
    ):
        meta = _resolve_meta(meta, connectivity, options)
        debug_print = bool(meta.get("options", {}).get("debug")) if meta.get("options") else False

        messages = _parse_history(context)
        if keep_context and not messages and self.saved_messages is not None:
            messages = [dict(message) for message in self.saved_messages]
        messages = _apply_system_message(messages, system)
        messages.append({"role": "user", "content": _images_to_content_parts(prompt, images)})

        payload = _build_request(meta["connectivity"]["model"], messages, think, format, meta.get("options"))
        if debug_print:
            _debug_request("llama.cpp generate", meta["connectivity"]["url"], payload)

        result, thinking, response = _run_chat(meta["connectivity"]["url"], payload)
        if debug_print:
            _debug_response("llama.cpp generate", response)

        messages.append({"role": "assistant", "content": result})
        if keep_context:
            self.saved_messages = [dict(message) for message in messages]

        return result, thinking if think else None, _history_to_string(messages), meta


class LlamaCppChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system": ("STRING", {"multiline": True, "default": "You are an AI artist."}),
                "prompt": ("STRING", {"multiline": True, "default": "What is art?"}),
                "think": ("BOOLEAN", {"default": False}),
                "format": (["text", "json"],),
            },
            "optional": {
                "connectivity": ("LLAMACPP_CONNECTIVITY", {"forceInput": False}),
                "options": ("LLAMACPP_OPTIONS", {"forceInput": False}),
                "images": ("IMAGE", {"forceInput": False}),
                "meta": ("LLAMACPP_META", {"forceInput": False}),
                "history": ("LLAMACPP_HISTORY", {"forceInput": False}),
                "reset_session": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "LLAMACPP_META", "LLAMACPP_HISTORY")
    RETURN_NAMES = ("result", "thinking", "meta", "history")
    FUNCTION = "llamacpp_chat"
    CATEGORY = CATEGORY
    DESCRIPTION = "Multi-turn chat through llama.cpp /v1/chat/completions."

    def llamacpp_chat(
        self,
        system: str,
        prompt: str,
        think: bool,
        unique_id: str,
        format: str,
        options: dict[str, Any] | None = None,
        connectivity: dict[str, Any] | None = None,
        images: list[torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
        history: str | None = None,
        reset_session: bool = False,
    ):
        meta = _resolve_meta(meta, connectivity, options)
        debug_print = bool(meta.get("options", {}).get("debug")) if meta.get("options") else False
        session_key = unique_id

        if history:
            messages = _parse_history(history)
        else:
            if reset_session or session_key not in CHAT_SESSIONS:
                CHAT_SESSIONS[session_key] = ChatSession()
            messages = [dict(message) for message in CHAT_SESSIONS[session_key].messages]

        if reset_session:
            messages = []

        messages = _apply_system_message(messages, system)
        messages.append({"role": "user", "content": _images_to_content_parts(prompt, images)})

        payload = _build_request(meta["connectivity"]["model"], messages, think, format, meta.get("options"))
        if debug_print:
            _debug_request("llama.cpp chat", meta["connectivity"]["url"], payload)

        result, thinking, response = _run_chat(meta["connectivity"]["url"], payload)
        if debug_print:
            _debug_response("llama.cpp chat", response)

        messages.append({"role": "assistant", "content": result})
        CHAT_SESSIONS[session_key] = ChatSession(messages=messages, model=meta["connectivity"]["model"])

        return result, thinking if think else None, meta, _history_to_string(messages)


def _resolve_meta(
    meta: dict[str, Any] | None,
    connectivity: dict[str, Any] | None,
    options: dict[str, Any] | None,
) -> dict[str, Any]:
    if meta is None:
        meta = {}
    else:
        meta = dict(meta)

    if connectivity is not None:
        meta["connectivity"] = connectivity
    if options is not None:
        meta["options"] = options
    elif "options" not in meta:
        meta["options"] = None

    if not meta.get("connectivity"):
        raise ValueError("Either connectivity or meta with connectivity must be provided.")
    return meta


def _apply_system_message(messages: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
    messages = [dict(message) for message in messages]
    if not system:
        return messages
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": system}
    else:
        messages.insert(0, {"role": "system", "content": system})
    return messages


def _debug_request(label: str, url: str, payload: dict[str, Any]) -> None:
    print(f"--- {label} request")
    print(f"url: {url}/v1/chat/completions")
    debug_payload = json.loads(json.dumps(payload, default=str))
    for message in debug_payload.get("messages", []):
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    part["image_url"] = {"url": "<base64 image>"}
    pprint(debug_payload)
    print("---------------------------------------------------------")


def _debug_response(label: str, response: dict[str, Any]) -> None:
    print(f"--- {label} response")
    pprint(response)
    print("---------------------------------------------------------")


NODE_CLASS_MAPPINGS = {
    "LlamaCppOptions": LlamaCppOptions,
    "LlamaCppConnectivity": LlamaCppConnectivity,
    "LlamaCppGenerate": LlamaCppGenerate,
    "LlamaCppSaveHistory": LlamaCppSaveHistory,
    "LlamaCppLoadHistory": LlamaCppLoadHistory,
    "LlamaCppChat": LlamaCppChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaCppOptions": "llama.cpp Options",
    "LlamaCppConnectivity": "llama.cpp Connectivity",
    "LlamaCppGenerate": "llama.cpp Generate",
    "LlamaCppSaveHistory": "llama.cpp Save History",
    "LlamaCppLoadHistory": "llama.cpp Load History",
    "LlamaCppChat": "llama.cpp Chat",
}
