from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import БАЗОВЫЕ_ВОПРОСЫ

pytestmark = [pytest.mark.smoke, pytest.mark.chroma]


class TestSmoke:
    def test_health(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.parametrize("question", БАЗОВЫЕ_ВОПРОСЫ[:5])
    def test_answer_not_empty(self, client: TestClient, question: str):
        resp = client.post("/api/ask", json={"question": question})
        assert resp.status_code == 200, f"{question}: {resp.text}"
        data = resp.json()
        assert data["question"] == question
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert isinstance(data["found_in_kb"], bool)
        assert isinstance(data["sources"], list)

    def test_answer_structure(self, client: TestClient):
        resp = client.post("/api/ask", json={"question": "Какие требования к СИЗ?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "found_in_kb" in data
        assert "retrieval_ms" in data
        assert "generation_ms" in data
        assert "total_ms" in data
        assert "llm_model" in data

    def test_source_structure(self, client: TestClient):
        resp = client.post("/api/ask", json={"question": "Какие виды инструктажей бывают?"})
        assert resp.status_code == 200
        data = resp.json()
        if data["found_in_kb"] and data["sources"]:
            src = data["sources"][0]
            assert "source_file" in src
            assert "page_number" in src
            assert "excerpt" in src

    def test_empty_question(self, client: TestClient):
        resp = client.post("/api/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_not_found_question(self, client: TestClient):
        resp = client.post("/api/ask", json={"question": "Какое завтра будет магнитное поле на Марсе?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["found_in_kb"] == False or "не найдена" in data["answer"].lower()

    def test_answer_has_russian(self, client: TestClient):
        resp = client.post("/api/ask", json={"question": "Какие меры безопасности при работе на высоте?"})
        assert resp.status_code == 200
        data = resp.json()
        if data["found_in_kb"]:
            import re
            russian = re.findall(r"[а-яёА-ЯЁ]", data["answer"])
            assert len(russian) > 10, "Ответ должен быть на русском языке"

    def test_all_fields_present(self, client: TestClient):
        resp = client.post("/api/ask", json={"question": "Что такое технологическая карта?"})
        assert resp.status_code == 200
        data = resp.json()
        expected = {"question", "answer", "sources", "found_in_kb",
                    "image_descriptions", "retrieval_ms", "generation_ms",
                    "total_ms", "llm_model"}
        assert expected.issubset(data.keys()), f"Missing fields: {expected - set(data.keys())}"
