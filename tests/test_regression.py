from __future__ import annotations

"""
Регрессионные тесты — полный прогон базовых сценариев.
Запуск: pytest tests/test_regression.py -v
"""

import pytest
from fastapi.testclient import TestClient

from conftest import БАЗОВЫЕ_ВОПРОСЫ

pytestmark = [pytest.mark.regression, pytest.mark.chroma]


class TestRegression:
    def test_all_basic_questions_return_ok(self, client: TestClient):
        http_errors = []
        not_found = []
        for q in БАЗОВЫЕ_ВОПРОСЫ:
            resp = client.post("/api/ask", json={"question": q})
            if resp.status_code != 200:
                http_errors.append(f"[{resp.status_code}] {q[:60]}")
            elif not resp.json().get("found_in_kb"):
                not_found.append(q[:60])
        assert not http_errors, "HTTP ошибки:\n" + "\n".join(http_errors)
        if not_found:
            print(f"\n⚠ NOT_FOUND ({len(not_found)}/{len(БАЗОВЫЕ_ВОПРОСЫ)}):")
            for q in not_found:
                print(f"  • {q}")

    def test_all_questions_have_valid_structure(self, client: TestClient):
        errors = []
        for q in БАЗОВЫЕ_ВОПРОСЫ:
            resp = client.post("/api/ask", json={"question": q})
            if resp.status_code != 200:
                continue
            data = resp.json()
            if not isinstance(data.get("answer"), str):
                errors.append(f"answer не строка: {q[:60]}")
            if not isinstance(data.get("sources"), list):
                errors.append(f"sources не список: {q[:60]}")
            if "total_ms" not in data:
                errors.append(f"нет total_ms: {q[:60]}")
        assert not errors, "\n".join(errors)

    def test_total_time_all_questions(self, client: TestClient):
        slow_questions = []
        for q in БАЗОВЫЕ_ВОПРОСЫ:
            resp = client.post("/api/ask", json={"question": q})
            if resp.status_code != 200:
                continue
            total = resp.json().get("total_ms")
            if total is None:
                continue
            if total > 25_000:
                slow_questions.append(f"{total}ms: {q[:60]}")
        if slow_questions:
            msg = "\n".join(slow_questions[:5])
            pytest.fail(f"Медленные запросы (>25s):\n{msg}")

    def test_config_after_questions(self, client: TestClient):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        config = resp.json()
        assert config["top_k"] == 5
        assert config["fetch_k"] == 15
        assert config["llm_model"]
