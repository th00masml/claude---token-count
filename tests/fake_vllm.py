#!/usr/bin/env python3
"""Stand-in for `vllm serve` used by the serve.sh integration test (stdlib only).

Accepts the vLLM command line, serves /health, /v1/chat/completions and /tokenize.
Answers with the gold SQL when the question is found in FAKE_VLLM_ANSWERS (JSON file
{question: sql}), else with a wrong query. With tools offered and FAKE_VLLM_EXPLORE=1
it first issues one run_sql tool call. FAKE_VLLM_FAIL=1 makes it die during startup;
FAKE_VLLM_PREEMPT_AFTER=n logs a vLLM-style preemption warning after n requests.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
assert args and args[0] == "serve", args
port = int(args[args.index("--port") + 1])
served = args[args.index("--served-model-name") + 1]
loras = []
if "--lora-modules" in args:
    i = args.index("--lora-modules") + 1
    while i < len(args) and not args[i].startswith("--"):
        loras.append(args[i].split("=", 1)[0])
        i += 1
print(f"INFO fake vllm starting {served} on {port}, loras={loras}, args={args}", flush=True)
if os.environ.get("FAKE_VLLM_FAIL"):
    print("ValueError: No available memory for the cache blocks.", flush=True)
    sys.exit(1)

answers = json.load(open(os.environ["FAKE_VLLM_ANSWERS"])) if os.environ.get("FAKE_VLLM_ANSWERS") else {}
preempt_after = int(os.environ.get("FAKE_VLLM_PREEMPT_AFTER", "0"))
state = {"n": 0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {})
        else:
            self._json(404, {})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._json(200, {"count": len(json.dumps(req.get("messages", ""))) // 4})
        if self.path != "/v1/chat/completions":
            return self._json(404, {})
        if req["model"] not in [served, *loras]:
            return self._json(404, {"error": f"model {req['model']} not served"})
        state["n"] += 1
        if preempt_after and state["n"] == preempt_after:
            print("WARNING scheduler.py: Sequence group 7 is preempted by PreemptionMode.RECOMPUTE", flush=True)
        msgs = req["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        has_tool_result = any(m["role"] == "tool" for m in msgs)
        msg = {"role": "assistant", "content": None}
        if req.get("tools") and os.environ.get("FAKE_VLLM_EXPLORE") and not has_tool_result:
            msg["tool_calls"] = [{"id": f"call{state['n']}", "type": "function", "function": {
                "name": "run_sql", "arguments": json.dumps({"query": "SELECT LIN_KOD FROM LIN"})}}]
            finish = "tool_calls"
        else:
            sql = next((s for q, s in answers.items() if q in user), "SELECT -1")
            msg["content"] = f"[{req['model']}]\n```sql\n{sql}\n```"
            finish = "stop"
        time.sleep(0.01)
        self._json(200, {"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                         "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                         "usage": {"prompt_tokens": len(user) // 4, "completion_tokens": 20,
                                   "total_tokens": len(user) // 4 + 20}})


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
