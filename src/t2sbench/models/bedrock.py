"""Amazon Bedrock via the Converse API."""

from __future__ import annotations

import re
import time

from t2sbench.models.base import GenParams, Generation, Message, Model, ModelError, ToolSpec


_CONTEXT_ERR = re.compile(r"too long|too many (input )?tokens|context (length|window)|max(imum)? tokens|input length",
                          re.IGNORECASE)


class BedrockModel(Model):
    backend = "bedrock"

    def __init__(self, name: str, model_id: str, region: str, client=None, native_tools: bool = True):
        self.name = name
        self.model_id = model_id
        self.region = region
        self.native_tools = native_tools
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                              read_timeout=300, connect_timeout=20),
            )
        self.client = client

    @staticmethod
    def _convert(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                role = "user"
                blocks = [{"toolResult": {"toolUseId": m.tool_call_id,
                                          "content": [{"text": m.content or "(empty)"}]}}]
            elif m.role == "assistant":
                role = "assistant"
                blocks = [{"text": m.content}] if m.content and m.content.strip() else []
                for tc in m.tool_calls or []:
                    blocks.append({"toolUse": {"toolUseId": tc["id"], "name": tc["name"],
                                               "input": tc.get("arguments") or {}}})
                if not blocks:
                    blocks = [{"text": "(no text)"}]
            else:
                role = "user"
                blocks = [{"text": m.content if m.content and m.content.strip() else "(empty)"}]
            # Converse requires alternating roles: merge consecutive messages of one role
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})
        return out

    def chat(self, messages, system=None, params=None, tools: list[ToolSpec] | None = None) -> Generation:
        p = params or GenParams()
        inference = {"maxTokens": p.max_tokens, "temperature": p.temperature}
        if p.top_p is not None:
            inference["topP"] = p.top_p
        if p.stop:
            inference["stopSequences"] = p.stop
        req = {"modelId": self.model_id, "messages": self._convert(messages), "inferenceConfig": inference}
        if system:
            req["system"] = [{"text": system}]
        uses_tools = bool(tools) or any(m.tool_calls or m.role == "tool" for m in messages)
        if uses_tools and tools:
            req["toolConfig"] = {"tools": [
                {"toolSpec": {"name": t.name, "description": t.description,
                              "inputSchema": {"json": t.parameters}}} for t in tools]}
        if p.extra:
            req["additionalModelRequestFields"] = p.extra
        t0 = time.perf_counter()
        try:
            resp = self.client.converse(**req)
        except Exception as e:  # botocore ClientError: ValidationException etc.
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            # Only a too-long prompt is a property of the question (-> recorded as no answer).
            # Any other ValidationException means our request is malformed: re-raise so the
            # runner does not record it and counts it towards its failure-rate stop.
            if code == "ValidationException" and _CONTEXT_ERR.search(str(e)):
                raise ModelError(f"{code}: {e}") from e
            raise
        latency = time.perf_counter() - t0
        texts, calls = [], []
        for block in resp.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                texts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                calls.append({"id": tu["toolUseId"], "name": tu["name"], "arguments": tu.get("input") or {}})
            # reasoningContent blocks (reasoning models) are ignored for the answer
        usage = resp.get("usage", {})
        return Generation(text="\n".join(texts), input_tokens=int(usage.get("inputTokens", 0)),
                          output_tokens=int(usage.get("outputTokens", 0)), latency_s=latency,
                          tool_calls=calls, stop_reason=resp.get("stopReason"))
