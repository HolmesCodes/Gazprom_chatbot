from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from factory_assistant.config import settings
from factory_assistant.ingest import ingest_documents
from factory_assistant.service import FactoryAssistantService


def describe_images_main() -> None:
    from factory_assistant.vision import describe_image

    images_dir = Path("data/images")
    if not images_dir.exists():
        print("Нет директории data/images/")
        return

    all_images = sorted(images_dir.rglob("*"))
    total = len(all_images)
    for i, path in enumerate(all_images, 1):
        if not path.is_file():
            continue
        print(f"[{i}/{total}] {path.relative_to(images_dir)}...", end=" ", flush=True)
        desc = describe_image(path, api_key=settings.llm_api_key, api_base=settings.llm_api_base_url)
        if desc:
            print(f"✓ {desc[:60]}...")
        else:
            print("✗ ошибка")


def ingest_main() -> None:
    parser = argparse.ArgumentParser(description="Индексация документов завода")
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=settings.documents_dir,
        help="Каталог с PDF/DOCX/TXT",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Пересоздать базу с нуля",
    )
    args = parser.parse_args()
    if args.recreate:
        chroma_dir = settings.chroma_dir.resolve()
        if chroma_dir.exists():
            import shutil
            shutil.rmtree(chroma_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        result = ingest_documents(
            documents_dir=args.documents_dir,
            recreate=True,
        )
    else:
        result = ingest_documents(
            documents_dir=args.documents_dir,
            recreate=False,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def serve_main() -> None:
    uvicorn.run(
        "factory_assistant.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Factory Assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Проиндексировать документы")
    ingest_parser.add_argument("--documents-dir", type=Path, default=settings.documents_dir)
    ingest_parser.add_argument("--recreate", action="store_true", help="Пересоздать базу с нуля")

    ask_parser = sub.add_parser("ask", help="Задать один вопрос RAG-боту")
    ask_parser.add_argument("question", nargs="+")

    onboard_parser = sub.add_parser("onboarding", help="Показать первый шаг адаптации")

    serve_parser = sub.add_parser("serve", help="Запустить HTTP API")
    serve_parser.add_argument("--host", default=settings.api_host)
    serve_parser.add_argument("--port", type=int, default=settings.api_port)

    sub.add_parser("describe-images", help="Пакетное описание всех картинок через Vision")

    args = parser.parse_args()

    if args.command == "ingest":
        if args.recreate:
            chroma_dir = settings.chroma_dir.resolve()
            if chroma_dir.exists():
                import shutil
                shutil.rmtree(chroma_dir)
            chroma_dir.mkdir(parents=True, exist_ok=True)
            result = ingest_documents(
                documents_dir=args.documents_dir,
                recreate=True,
            )
        else:
            result = ingest_documents(
                documents_dir=args.documents_dir,
                recreate=False,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "ask":
        question = " ".join(args.question)
        svc = FactoryAssistantService()
        print(json.dumps(svc.ask(question), ensure_ascii=False, indent=2))
        return

    if args.command == "onboarding":
        svc = FactoryAssistantService()
        print(json.dumps(svc.start_onboarding(), ensure_ascii=False, indent=2))
        return

    if args.command == "serve":
        uvicorn.run(
            "factory_assistant.api:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return

    if args.command == "describe-images":
        describe_images_main()
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
