"""Sub-agents для улучшения RAG pipeline."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

REFORMULATE_SYSTEM = """\
Ты — агент переформулирования поисковых запросов.
Задача: переписать вопрос сотрудника так, чтобы он был максимально понятен для поиска по документации.

Правила:
1. Сохрани смысл вопроса.
2. Замени разговорные формулировки на термины из документации.
3. Добавь ключевые слова, которые могут встретиться в документах.
4. Длина переформулированного запроса: 10-30 слов.
5. Верни ТОЛЬКО переформулированный текст, без кавычек и пояснений."""


class ReformulationAgent:
    """Переформулирует вопрос для лучшего поиска в vector store."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def reformulate(self, question: str) -> str:
        try:
            resp = self.llm.invoke([
                ("system", REFORMULATE_SYSTEM),
                ("human", question),
            ])
            result = resp.content.strip()
            if len(result) > 10:
                logger.info("Reformulation: '%s' → '%s'", question, result)
                return result
        except Exception as exc:
            logger.warning("Reformulation failed: %s", exc)
        return question


EXTRACT_RELEVANT_IMAGES = """\
Ты — агент определения релевантности изображений.

Дан ответ на вопрос сотрудника и список описаний изображений со страниц документа.

Ответ:
{answer}

Описания изображений:
{image_descriptions}

Задача: определи, какие изображения РЕЛЕВАНТНЫ ответу.
Верни JSON-массив имён файлов релевантных изображений.
Если ни одно изображение не релевантно — верни пустой массив [].

Примеры:
- Вопрос "как выглядит пандус?", ответ описывает пандус → ["image1.png"]
- Вопрос "размеры поля?", ответ содержит цифры → ["image3.png", "image7.png"]
- Вопрос "что такое маркировка?", нет картинок в ответе → []

Верни ТОЛЬКО JSON-массив, без пояснений."""


class ImageRelevanceAgent:
    """Определяет какие картинки релевантны ответу LLM."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def filter_images(
        self,
        answer: str,
        sources_with_images: list[dict],
    ) -> list[str]:
        """Фильтрует image_paths по релевантности к ответу.

        Args:
            answer: ответ LLM
            sources_with_images: [{"source_file": ..., "page_number": ..., "image_paths": [...], "image_descriptions": {...}}]

        Returns:
            Список релевантных имён файлов картинок (уникальных).
        """
        all_images = {}
        descriptions_map = {}
        for src in sources_with_images:
            for img in src.get("image_paths", []):
                all_images[img] = src.get("source_file", "")
            for img, desc in src.get("image_descriptions", {}).items():
                descriptions_map[img] = desc

        if not all_images:
            return []

        desc_lines = []
        for img, source in all_images.items():
            desc = descriptions_map.get(img, "описание отсутствует")
            desc_lines.append(f"- {img} ({source}): {desc}")
        image_descriptions_text = "\n".join(desc_lines)

        try:
            prompt = EXTRACT_RELEVANT_IMAGES.format(
                answer=answer,
                image_descriptions=image_descriptions_text,
            )
            resp = self.llm.invoke([("human", prompt)])
            text = resp.content.strip()

            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                relevant = json.loads(match.group())
                if isinstance(relevant, list):
                    valid = [f for f in relevant if f in all_images]
                    logger.info("Image relevance: %d/%d images relevant", len(valid), len(all_images))
                    return valid

        except Exception as exc:
            logger.warning("Image relevance failed: %s", exc)

        return list(all_images.keys())
