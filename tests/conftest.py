from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from factory_assistant.api import app, service
from factory_assistant.config import Settings, settings
from factory_assistant.service import FactoryAssistantService


@pytest.fixture(scope="session")
def cfg() -> Settings:
    return settings


@pytest.fixture(scope="session")
def client(cfg: Settings) -> TestClient:
    from factory_assistant.api import service as global_service
    svc = FactoryAssistantService()
    global_service = svc
    import factory_assistant.api
    factory_assistant.api.service = svc
    client_ = TestClient(app)
    yield client_


@pytest.fixture(scope="function")
def svc() -> FactoryAssistantService:
    return FactoryAssistantService()


CHROMA_READY = None


def chroma_available() -> bool:
    global CHROMA_READY
    if CHROMA_READY is not None:
        return CHROMA_READY
    chroma_dir = settings.chroma_dir.resolve()
    ok = chroma_dir.exists() and any(chroma_dir.iterdir())
    CHROMA_READY = ok
    return ok


def pytest_collection_modifyitems(config, items):
    for item in items:
        if not chroma_available():
            if "chroma" in item.keywords or "search" in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason="ChromaDB не найдена. Сначала выполните индексацию: python -m factory_assistant.cli ingest")
                )


БАЗОВЫЕ_ВОПРОСЫ = [
    "Какие меры безопасности при работе на высоте?",
    "Что делать при пожаре на производстве?",
    "Как часто проводится ТО оборудования?",
    "Какие требования к строповке грузов?",
    "Какая ответственность за нарушение промбезопасности?",
    "Какие марки стали используются для сварных конструкций?",
    "Как проводится адаптация нового сотрудника?",
    "Что такое технологическая карта?",
    "Какие бывают виды инструктажей?",
    "Какие требования к СИЗ?",
    "Как проводится расследование инцидентов?",
    "Какие параметры контролируются при эксплуатации насосов?",
    "Что входит в ежесменное обслуживание оборудования?",
    "Какие требования к заземлению оборудования?",
    "Какие документы регламентируют промышленную безопасность?",
]
