import pytest

from t2sbench.datasets.synthetic.build import build_all


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    """Synthetic DB + question sets built once per test session."""
    out = tmp_path_factory.mktemp("synthetic")
    manifest = build_all(out, seed=42)
    return out, manifest
