from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

pytestmark = [pytest.mark.config]


class TestConfig:
    def test_get_config(self, client: TestClient):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_model" in data
        assert "top_k" in data
        assert "fetch_k" in data
        assert "relevance_threshold" in data
        assert "chunk_size" in data
        assert "chunk_overlap" in data
        assert data["top_k"] == 5
        assert data["fetch_k"] == 15

    def test_patch_config_top_k(self, client: TestClient):
        old = client.get("/api/config").json()
        old_top_k = old["top_k"]
        new_val = 7
        resp = client.patch("/api/admin/config", json={"top_k": new_val})
        assert resp.status_code == 200
        data = resp.json()
        assert data["top_k"] == new_val
        # Возвращаем обратно
        client.patch("/api/admin/config", json={"top_k": old_top_k})
        restored = client.get("/api/config").json()
        assert restored["top_k"] == old_top_k

    def test_patch_config_chunk_size(self, client: TestClient):
        old = client.get("/api/config").json()
        old_chunk = old.get("chunk_size", 1500)
        resp = client.patch("/api/admin/config", json={"chunk_size": 1000})
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_size"] == 1000
        # Возвращаем обратно
        client.patch("/api/admin/config", json={"chunk_size": old_chunk})
        restored = client.get("/api/config").json()
        assert restored["chunk_size"] == old_chunk

    def test_patch_config_full(self, client: TestClient):
        old = client.get("/api/config").json()
        patch = {
            "top_k": 3,
            "fetch_k": 10,
            "relevance_threshold": -1.0,
        }
        resp = client.patch("/api/admin/config", json=patch)
        assert resp.status_code == 200
        data = resp.json()
        for k, v in patch.items():
            assert data[k] == v, f"{k}: ожидалось {v}, получено {data[k]}"
        # Восстановление
        client.patch("/api/admin/config", json={
            "top_k": old["top_k"],
            "fetch_k": old["fetch_k"],
            "relevance_threshold": old["relevance_threshold"],
        })

    def test_patch_config_invalid_values(self, client: TestClient):
        resp = client.patch("/api/admin/config", json={"top_k": -1})
        assert resp.status_code == 422
        resp = client.patch("/api/admin/config", json={"chunk_size": 50})
        assert resp.status_code == 422
        resp = client.patch("/api/admin/config", json={"chunk_overlap": 2000})
        assert resp.status_code == 422

    def test_health_endpoint(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_index_status_endpoint(self, client: TestClient):
        resp = client.get("/api/admin/index/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "chroma_ready" in data
        assert "documents_count" in data
        assert "indexed_chunks" in data
        assert "files" in data
