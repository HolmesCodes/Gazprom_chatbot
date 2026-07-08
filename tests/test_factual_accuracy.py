from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import БАЗОВЫЕ_ВОПРОСЫ

pytestmark = [pytest.mark.chroma, pytest.mark.regression]

# Ключевые слова, которые должны быть в ответе для каждого вопроса
# (грубая проверка достоверности — LLM не должна игнорировать контекст)
ЭТАЛОННЫЕ_КЛЮЧИ: dict[str, list[str]] = {
    "Какие меры безопасности при работе на высоте?": [
        "страховочн", "привяз", "высот", "наряд-допуск",
    ],
    "Что делать при пожаре на производстве?": [
        "пожар", "эвакуац", "тушен",
    ],
    "Какая ответственность за нарушение промбезопасности?": [
        "ответствен", "нарушен", "промышлен",
    ],
    "Какие требования к строповке грузов?": [
        "строп", "груз", "ветв",
    ],
}


class TestFactualAccuracy:
    @pytest.mark.parametrize("question,keywords", list(ЭТАЛОННЫЕ_КЛЮЧИ.items()))
    def test_answer_contains_keywords(self, client: TestClient, question: str, keywords: list[str]):
        resp = client.post("/api/ask", json={"question": question})
        assert resp.status_code == 200
        data = resp.json()
        if not data["found_in_kb"]:
            pytest.skip("Информация не найдена в БЗ — проверка ключевых слов пропущена")
        answer_lower = data["answer"].lower()
        missing = [kw for kw in keywords if kw.lower() not in answer_lower]
        assert not missing, (
            f"В ответе на '{question}' отсутствуют ключевые слова: {missing}"
        )

    def test_answer_cites_sources(self, client: TestClient):
        resp = client.post("/api/ask", json={"question": "Какие виды инструктажей бывают?"})
        assert resp.status_code == 200
        data = resp.json()
        if data["found_in_kb"] and data["sources"]:
            answer = data["answer"].lower()
            has_citation = any(
                src["source_file"].lower() in answer
                for src in data["sources"]
            )
            has_page = any(
                f"стр" in answer
                for src in data["sources"]
                if src.get("page_number")
            )
            assert has_citation or has_page, (
                "Ответ не ссылается на источники (ни имя файла, ни страница)"
            )

    def test_no_hallucination_on_invented_query(self, client: TestClient):
        resp = client.post("/api/ask", json={
            "question": "Какие требования к оборудованию для квантовой телепортации на заводе?"
        })
        assert resp.status_code == 200
        data = resp.json()
        if data["found_in_kb"]:
            # Если нашлось — хотя бы проверяем, что ответ не слишком уверенный
            answer = data["answer"].lower()
            assert "квантов" not in answer or "не" in answer, (
                "Ответ не должен утверждать наличие документов по квантовой телепортации"
            )

    def test_sources_match_excerpt(self, client: TestClient):
        resp = client.post("/api/ask", json={"question": "Какие марки стали используются?"})
        assert resp.status_code == 200
        data = resp.json()
        if data["found_in_kb"] and data["sources"]:
            for src in data["sources"]:
                assert "source_file" in src
                assert src["source_file"], "source_file не должен быть пустым"
                if src.get("excerpt"):
                    assert len(src["excerpt"]) > 10, "excerpt слишком короткий"
