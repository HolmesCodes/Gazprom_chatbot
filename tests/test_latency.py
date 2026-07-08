from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import БАЗОВЫЕ_ВОПРОСЫ

pytestmark = [pytest.mark.slow, pytest.mark.chroma]

MAX_TOTAL_MS = 30_000
MAX_GENERATION_MS = 25_000


class TestLatency:
    @pytest.mark.parametrize("question", БАЗОВЫЕ_ВОПРОСЫ[:3])
    def test_response_under_timeout(self, client: TestClient, question: str):
        resp = client.post("/api/ask", json={"question": question})
        assert resp.status_code == 200
        data = resp.json()
        total = data.get("total_ms")
        if total is None:
            pytest.skip("total_ms is None (NOT_FOUND)")
        assert total < MAX_TOTAL_MS, (
            f"Слишком долгий ответ на '{question[:40]}': {total}ms (лимит {MAX_TOTAL_MS}ms)"
        )

    def test_generation_not_extreme(self, client: TestClient):
        total_gen = 0
        count = 0
        for q in БАЗОВЫЕ_ВОПРОСЫ[:5]:
            resp = client.post("/api/ask", json={"question": q})
            assert resp.status_code == 200
            data = resp.json()
            if data["found_in_kb"]:
                gen = data.get("generation_ms", 0)
                total_gen += gen
                count += 1
        if count:
            avg = total_gen / count
            assert avg < MAX_GENERATION_MS, (
                f"Среднее время генерации: {avg:.0f}ms (лимит {MAX_GENERATION_MS}ms)"
            )

    def test_retrieval_is_fast(self, client: TestClient):
        for q in БАЗОВЫЕ_ВОПРОСЫ[:3]:
            resp = client.post("/api/ask", json={"question": q})
            assert resp.status_code == 200
            data = resp.json()
            retrieval = data.get("retrieval_ms")
            if retrieval is None:
                continue
            assert retrieval < 5000, (
                f"Поиск слишком долгий для '{q[:40]}': {retrieval}ms"
            )

    def test_consecutive_questions_stable(self, client: TestClient):
        times = []
        for q in БАЗОВЫЕ_ВОПРОСЫ[:5]:
            resp = client.post("/api/ask", json={"question": q})
            assert resp.status_code == 200
            t = resp.json().get("total_ms")
            if t is not None:
                times.append(t)
        if len(times) < 2:
            pytest.skip("Недостаточно ответов с замером времени")
        max_t = max(times)
        min_t = min(times)
        assert max_t - min_t < 15_000, (
            f"Разброс времени слишком большой: от {min_t}ms до {max_t}ms"
        )
