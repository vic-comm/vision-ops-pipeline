import subprocess
import sys
from pathlib import Path
import time

import pytest

from fastapi.testclient import TestClient


def ensure_demo_model():
    artifacts = Path("artifacts")
    model_file = artifacts / "efficientnet_b3_model.pth"
    thresh_file = artifacts / "efficientnet_b3_threshold.pkl"
    if model_file.exists() and thresh_file.exists():
        return

    # Run demo training to create artifacts
    try:
        subprocess.check_call([sys.executable, "examples/demo_train.py"]) 
    except subprocess.CalledProcessError:
        pytest.skip("Could not build demo model in this environment")


def test_health_and_model_info():
    try:
        import torch
    except Exception:
        pytest.skip("torch not installed in this environment; skipping smoke test")

    ensure_demo_model()

    # import here so artifacts exist before app startup
    from api_service.app import app

    client = TestClient(app)

    # Wait briefly for lifespan startup
    time.sleep(1)

    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"

    r2 = client.get("/model/info")
    assert r2.status_code == 200
    info = r2.json()
    assert "model_path" in info
