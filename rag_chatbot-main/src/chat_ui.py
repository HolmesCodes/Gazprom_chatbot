import os
import sys
import time
from pathlib import Path

# macOS OpenMP can load duplicate libomp copies from different libraries.
# This unsafe flag allows the program to continue instead of aborting.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from llm_rag import LLMRAGHandler
from conversation import ConversationManager

# Folder for saving uploaded PDFs
UPLOAD_DIR = Path("uploaded_pdfs")
UPLOAD_DIR.mkdir(exist_ok=True)

MENU_PROMPT = (
    "\nВыберите действие:\n"
    "1) Задать вопрос к PDF\n"
    "2) Добавить новые PDF из папки uploaded_pdfs\n"
    "3) Обновить / перестроить векторную базу из uploaded_pdfs\n"
    "4) Показать сохранённые PDF\n"
    "5) Сбросить разговор\n"
    "6) Выйти\n"
    "Ввод: "
)


def animate(message: str, duration: float = 0.8, steps: int = 3) -> None:
    for i in range(steps):
        print(f"{message}{'.' * ((i % 3) + 1)}", end="\r", flush=True)
        time.sleep(duration / steps)
    print(" " * (len(message) + 5), end="\r")


def header() -> None:
    print("=" * 60)
    print("RAG Chatbot (CLI) — Поиск ответов в PDF с указанием ссылок на источник")
    print("Папка для документов:", UPLOAD_DIR.resolve())
    print("=" * 60)


def list_pdf_files() -> list[Path]:
    return sorted(UPLOAD_DIR.glob("*.pdf"))


def show_indexed_files(processed_files: set[str]) -> None:
    if not processed_files:
        print("Нет найденных PDF в папке uploaded_pdfs.")
        return
    print("Найденные PDF:")
    for file_name in sorted(processed_files):
        print(f"- {file_name}")


def add_new_pdfs(handler: LLMRAGHandler, processed_files: set[str]) -> None:
    files = [pdf for pdf in list_pdf_files() if pdf.name not in processed_files]
    if not files:
        print("Новых PDF не найдено. Добавьте файлы в папку uploaded_pdfs и попробуйте снова.")
        return

    for pdf_file in files:
        print(f"Добавляю и индексирую: {pdf_file.name}")
        animate("Обработка PDF")
        handler.add_pdf_to_context(pdf_file)
        processed_files.add(pdf_file.name)
        print(f"[OK] {pdf_file.name} добавлен в векторную базу.")


def rebuild_vector_db(handler: LLMRAGHandler, processed_files: set[str]) -> None:
    print("Перестройка векторной базы из uploaded_pdfs...")
    animate("Перестройка базы")
    handler.vector_store.rebuild_index(str(UPLOAD_DIR))
    processed_files.clear()
    processed_files.update({pdf.name for pdf in list_pdf_files()})
    print("Векторная база обновлена.")


def ask_question(handler: LLMRAGHandler, conversation_manager: ConversationManager) -> None:
    question = input("Введите вопрос: ").strip()
    if not question:
        print("Вопрос не должен быть пустым.")
        return
    animate("Генерация ответа")
    try:
        answer = handler.generate_response(question)
    except Exception as exc:
        print(f"Ошибка при запросе к Gemini API: {exc}")
        return
    print("\nОтвет:\n" + "-" * 60)
    print(answer)
    print("-" * 60)
    conversation_manager.save(handler.get_history())


def reset_conversation(handler: LLMRAGHandler, conversation_manager: ConversationManager) -> None:
    handler.reset()
    conversation_manager.clear()
    print("Разговор сброшен.")


def main() -> None:
    conversation_manager = ConversationManager()
    handler = LLMRAGHandler()
    saved_conversation = conversation_manager.load()
    if saved_conversation:
        handler.history = saved_conversation
    processed_files = {pdf.name for pdf in list_pdf_files()}

    header()
    while True:
        choice = input(MENU_PROMPT).strip()
        if choice == "1":
            ask_question(handler, conversation_manager)
        elif choice == "2":
            add_new_pdfs(handler, processed_files)
        elif choice == "3":
            rebuild_vector_db(handler, processed_files)
        elif choice == "4":
            show_indexed_files(processed_files)
        elif choice == "5":
            reset_conversation(handler, conversation_manager)
        elif choice == "6":
            print("Выход. Пока!")
            sys.exit(0)
        else:
            print("Неверный ввод. Пожалуйста, выберите цифру от 1 до 6.")


if __name__ == "__main__":
    main()
