from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from factory_assistant.rag import RagEngine
from factory_assistant.config import settings

pytestmark = [pytest.mark.search, pytest.mark.chroma]

# Ground truth: для каждого вопроса — какие файлы/слова должны быть в выдаче.
# Заполняется по мере аннотирования данных.
GROUND_TRUTH: dict[str, set[str]] = {
    "Какие меры безопасности при работе на высоте?": {
        "охрана труда", "высот", "страховк",
    },
    "Какая ответственность за нарушение промбезопасности?": {
        "промышлен", "ответствен", "безопасност",
    },
    "Как часто проводится ТО оборудования?": {
        "то", "обслуживан", "ежесмен", "еженедель", "ежемесячн",
    },
    "Какие требования к строповке грузов?": {
        "строп", "грузоподъемн", "канатн",
    },
}


class TestSearchQuality:
    @pytest.fixture(scope="class")
    def rag(self) -> RagEngine:
        engine = RagEngine(settings)
        assert engine.vectorstore is not None, "ChromaDB не загружена"
        return engine

    def test_mrr(self, rag: RagEngine):
        reciprocal_ranks = []
        for question, expected in GROUND_TRUTH.items():
            docs = rag._retrieve(question)
            if not docs:
                continue
            rank = None
            for i, doc in enumerate(docs):
                content = doc.page_content.lower()
                if any(kw.lower() in content for kw in expected):
                    rank = i + 1
                    break
            if rank:
                reciprocal_ranks.append(1.0 / rank)
        assert len(reciprocal_ranks) > 0, "Ни один запрос не нашёл релевантные документы"
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        assert mrr > 0.2, f"MRR слишком низкий: {mrr:.3f} (нужно > 0.2)"

    @pytest.mark.skip(reason="Требуется ручная аннотация ground truth для documents")
    def test_recall_at_3(self, rag: RagEngine):
        recalls = []
        for question, expected in GROUND_TRUTH.items():
            docs = rag._retrieve(question)
            if not docs:
                continue
            top3 = docs[:3]
            found_keywords = set()
            for doc in top3:
                content = doc.page_content.lower()
                for kw in expected:
                    if kw.lower() in content:
                        found_keywords.add(kw)
            recall = len(found_keywords) / len(expected) if expected else 0
            recalls.append(recall)
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        assert avg_recall > 0.15, f"Средний Recall@3: {avg_recall:.3f} (нужно > 0.15)"

    def test_precision_at_3(self, rag: RagEngine):
        precisions = []
        for question, expected in GROUND_TRUTH.items():
            docs = rag._retrieve(question)
            if not docs:
                continue
            top3 = docs[:3]
            relevant = 0
            for doc in top3:
                content = doc.page_content.lower()
                if any(kw.lower() in content for kw in expected):
                    relevant += 1
            precision = relevant / min(len(top3), 3)
            precisions.append(precision)
        avg_precision = sum(precisions) / len(precisions) if precisions else 0
        assert avg_precision > 0.3, f"Средний Precision@3: {avg_precision:.3f} (нужно > 0.3)"

    def test_document_coverage(self, rag: RagEngine):
        all_retrieved_files: set[str] = set()
        for question in GROUND_TRUTH:
            docs = rag._retrieve(question)
            for doc in docs:
                fname = doc.metadata.get("source_file", "")
                if fname:
                    all_retrieved_files.add(fname)
        assert len(all_retrieved_files) > 0, "Не найдено ни одного файла"
        print(f"\nЗатронуто файлов: {len(all_retrieved_files)}")

    def test_consistent_retrieval(self, rag: RagEngine):
        for question in list(GROUND_TRUTH.keys())[:2]:
            results1 = rag._retrieve(question)
            results2 = rag._retrieve(question)
            urls1 = [d.metadata.get("source_file", "") + str(d.metadata.get("page_number", ""))
                     for d in results1]
            urls2 = [d.metadata.get("source_file", "") + str(d.metadata.get("page_number", ""))
                     for d in results2]
            overlap = set(urls1) & set(urls2)
            min_count = min(len(urls1), len(urls2))
            assert len(overlap) >= max(min_count * 0.4, 1), (
                f"Нестабильный поиск для '{question[:40]}': найдено {len(urls1)} и {len(urls2)} чанков, "
                f"пересечение {len(overlap)}"
            )

    def test_retrieval_time(self, rag: RagEngine):
        import time
        times = []
        for question in GROUND_TRUTH:
            t0 = time.perf_counter()
            rag._retrieve(question)
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed)
        avg = sum(times) / len(times)
        assert avg < 3000, f"Среднее время поиска: {avg:.0f}ms (лимит 3000ms)"
