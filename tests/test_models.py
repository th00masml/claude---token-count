import json
from types import SimpleNamespace

import pytest

from fakes import FakeBedrockClient, FakeModel, fake_openai_client
from t2sbench.models.base import Generation, GenParams, Message, ModelError, ToolSpec
from t2sbench.models.bedrock import BedrockModel
from t2sbench.models.budget import Budget, BudgetExceeded, UnpricedModel
from t2sbench.models.cache import DiskCache
from t2sbench.models.cached import CachedModel
from t2sbench.models.openai_compat import OpenAICompatModel

TOOL = ToolSpec("run_sql", "run", {"type": "object", "properties": {"query": {"type": "string"}}})


@pytest.fixture()
def prices(tmp_path):
    p = tmp_path / "prices.yaml"
    p.write_text("per_million_tokens:\n  cheap: {input: 1.0, output: 2.0}\n  nopr: {input: null, output: null}\n"
                 "budget_usd: 0.001\n")
    return p


def test_bedrock_message_conversion_merges_tool_results():
    msgs = [Message("user", "q"),
            Message("assistant", "", tool_calls=[{"id": "a", "name": "run_sql", "arguments": {"query": "SELECT 1"}},
                                                 {"id": "b", "name": "run_sql", "arguments": {"query": "SELECT 2"}}]),
            Message("tool", "1", tool_call_id="a"), Message("tool", "2", tool_call_id="b")]
    out = BedrockModel._convert(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert len(out[2]["content"]) == 2 and out[2]["content"][0]["toolResult"]["toolUseId"] == "a"
    assert out[1]["content"][0]["toolUse"]["input"] == {"query": "SELECT 1"}


def test_bedrock_chat_parses_text_tools_and_usage():
    resp = {"output": {"message": {"content": [{"reasoningContent": {"x": 1}}, {"text": "thinking"},
                                               {"toolUse": {"toolUseId": "t1", "name": "run_sql", "input": {"query": "SELECT 1"}}}]}},
            "usage": {"inputTokens": 50, "outputTokens": 9}, "stopReason": "tool_use"}
    client = FakeBedrockClient([resp])
    m = BedrockModel("cheap", "amazon.x", "us-east-1", client=client)
    g = m.chat([Message("user", "hi")], system="sys", params=GenParams(temperature=0.0, max_tokens=10), tools=[TOOL])
    assert g.text == "thinking" and g.tool_calls[0]["id"] == "t1" and g.input_tokens == 50
    req = client.requests[0]
    assert req["system"] == [{"text": "sys"}] and "toolConfig" in req and "topP" not in req["inferenceConfig"]


def test_bedrock_validation_error_is_model_error():
    err = Exception("bad")
    err.response = {"Error": {"Code": "ValidationException"}}
    m = BedrockModel("cheap", "amazon.x", "us-east-1", client=FakeBedrockClient([err]))
    with pytest.raises(ModelError):
        m.chat([Message("user", "hi")])


def test_openai_compat_uses_adapter_name_and_parses_tools():
    tc = [SimpleNamespace(id="c1", function=SimpleNamespace(name="run_sql", arguments=json.dumps({"query": "SELECT 1"})))]
    client, calls = fake_openai_client(text=None, tool_calls=tc)
    m = OpenAICompatModel("qwen", "http://x", adapter="qwen-L2-400", client=client)
    g = m.chat([Message("user", "hi")], system="s", params=GenParams(seed=3, extra={"top_k": 20}), tools=[TOOL])
    assert calls[0]["model"] == "qwen-L2-400" and calls[0]["seed"] == 3 and calls[0]["extra_body"] == {"top_k": 20}
    assert calls[0]["messages"][0] == {"role": "system", "content": "s"}
    assert g.tool_calls == [{"id": "c1", "name": "run_sql", "arguments": {"query": "SELECT 1"}}] and g.text == ""


def test_cache_replays_and_budget_counts_once(tmp_path, prices):
    inner = FakeModel(lambda *a: Generation("```sql\nSELECT 1\n```", 100, 10, 0.5), name="cheap", backend="bedrock")
    b = Budget(prices, tmp_path / "spend.json", budget_usd=1.0)
    m = CachedModel(inner, DiskCache(tmp_path / "c"), b, identity="amazon.x")
    g1 = m.generate([Message("user", "q")])
    g2 = m.generate([Message("user", "q")])
    assert len(inner.calls) == 1 and not g1.cached and g2.cached and g2.latency_s == 0.5
    assert b.state["by_model"]["cheap"]["calls"] == 1
    assert json.loads((tmp_path / "spend.json").read_text())["total_usd"] == pytest.approx((100 * 1 + 10 * 2) / 1e6)


def test_budget_stops_before_call(tmp_path, prices):
    inner = FakeModel(lambda *a: "x", name="cheap", backend="bedrock")
    m = CachedModel(inner, DiskCache(tmp_path / "c"), Budget(prices, tmp_path / "s.json"))  # budget 0.001 USD
    with pytest.raises(BudgetExceeded):
        m.generate([Message("user", "q" * 3000)], params=GenParams(max_tokens=1000))
    assert inner.calls == []


def test_unpriced_bedrock_model_refused(tmp_path, prices):
    inner = FakeModel(lambda *a: "x", name="nopr", backend="bedrock")
    m = CachedModel(inner, DiskCache(tmp_path / "c"), Budget(prices, tmp_path / "s.json"))
    with pytest.raises(UnpricedModel):
        m.generate([Message("user", "q")])


def test_local_model_ignores_budget(tmp_path, prices):
    inner = FakeModel(lambda *a: "x", name="nopr", backend="vllm")
    m = CachedModel(inner, DiskCache(tmp_path / "c"), Budget(prices, tmp_path / "s.json"))
    assert m.generate([Message("user", "q")]).text == "x"


def _tool_caller(n_calls_before_answer):
    state = {"i": 0}

    def respond(messages, system, params, tools):
        state["i"] += 1
        if state["i"] <= n_calls_before_answer:
            return Generation("", 10, 2, 0.1, tool_calls=[{"id": f"t{state['i']}", "name": "run_sql",
                                                         "arguments": {"query": "SELECT 1"}}])
        return Generation("```sql\nSELECT 2\n```", 10, 2, 0.1)
    return respond


def test_native_tool_loop_enforces_limit():
    handled = []
    m = FakeModel(_tool_caller(3))
    g, transcript = m.generate_with_tools([Message("user", "q")], [TOOL], lambda n, a: handled.append(a) or "rows",
                                          max_tool_calls=2)
    assert len(handled) == 2
    assert g.text.endswith("```") and g.n_calls == 4 and g.input_tokens == 40
    tool_msgs = [x for x in transcript if x.role == "tool"]
    assert "limit reached" in tool_msgs[-1].content.lower()


def test_native_tool_loop_gives_up_without_answer():
    m = FakeModel(_tool_caller(99))
    g, _ = m.generate_with_tools([Message("user", "q")], [TOOL], lambda n, a: "rows", max_tool_calls=2)
    assert g.text == "" and g.n_calls == 4


def test_text_tool_protocol():
    replies = iter(["<explore>SELECT DISTINCT a FROM t</explore>", "```sql\nSELECT a FROM t\n```"])
    m = FakeModel(lambda *a: next(replies), native_tools=False)
    seen = []
    g, transcript = m.generate_with_tools([Message("user", "q")], [TOOL], lambda n, a: seen.append(a["query"]) or "a\n1")
    assert seen == ["SELECT DISTINCT a FROM t"] and "SELECT a FROM t" in g.text
    assert "<explore>" in m.calls[0]["system"]
    assert transcript[2].role == "user" and "Result" in transcript[2].content
