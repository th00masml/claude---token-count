from t2sbench.models.base import GenParams, Message
from t2sbench.models.cache import DiskCache, cache_key


def test_key_is_stable_and_sensitive():
    msgs = [Message("user", "hi")]
    k1 = cache_key("m", None, msgs, GenParams(temperature=0.0))
    assert k1 == cache_key("m", None, [Message("user", "hi")], GenParams(temperature=0.0))
    assert k1 != cache_key("m", "lora-a", msgs, GenParams(temperature=0.0))
    assert k1 != cache_key("m2", None, msgs, GenParams(temperature=0.0))
    assert k1 != cache_key("m", None, msgs, GenParams(temperature=0.7))
    assert k1 != cache_key("m", None, [Message("user", "hi!")], GenParams(temperature=0.0))


def test_roundtrip(tmp_path):
    c = DiskCache(tmp_path)
    k = cache_key("m", None, "p", {})
    assert c.get(k) is None
    c.put(k, {"text": "SELECT 1", "input_tokens": 3})
    assert c.get(k) == {"text": "SELECT 1", "input_tokens": 3}
