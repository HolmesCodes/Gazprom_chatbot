from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from factory_assistant.config import Settings, settings
from factory_assistant.ingest import load_documents
from factory_assistant.prompts import ONBOARDING_SYSTEM_PROMPT

STEP_PATTERN = re.compile(
    r"^(?:Шаг\s*)?(\d+)[.)]\s*(.+?)(?=^(?:Шаг\s*)?\d+[.)]|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass
class OnboardingStep:
    number: int
    title: str
    body: str

    def as_dict(self) -> dict:
        return {"number": self.number, "title": self.title, "body": self.body}


@dataclass
class OnboardingSession:
    session_id: str
    steps: list[OnboardingStep]
    current_index: int = 0
    history: list[dict] = field(default_factory=list)

    @property
    def current_step(self) -> OnboardingStep:
        return self.steps[self.current_index]

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    def as_dict(self) -> dict:
        step = self.current_step
        return {
            "session_id": self.session_id,
            "current_step": step.number,
            "total_steps": self.total_steps,
            "title": step.title,
            "body": step.body,
            "history_length": len(self.history),
        }


def parse_onboarding_steps(text: str) -> list[OnboardingStep]:
    matches = list(STEP_PATTERN.finditer(text.strip()))
    if not matches:
        raise ValueError("Не удалось распарсить шаги адаптационного чек-листа.")

    steps: list[OnboardingStep] = []
    for match in matches:
        number = int(match.group(1))
        block = match.group(2).strip()
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        title = lines[0] if lines else f"Шаг {number}"
        body = "\n".join(lines[1:]) if len(lines) > 1 else block
        steps.append(OnboardingStep(number=number, title=title, body=body))
    return steps


def load_onboarding_steps(cfg: Settings | None = None) -> list[OnboardingStep]:
    cfg = cfg or settings
    doc_path = (cfg.documents_dir / cfg.onboarding_doc_name).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"Чек-лист не найден: {doc_path}")

    documents = load_documents(cfg.documents_dir)
    checklist_docs = [
        doc for doc in documents if doc.metadata.get("source_file") == cfg.onboarding_doc_name
    ]
    if not checklist_docs:
        text = doc_path.read_text(encoding="utf-8")
    else:
        text = "\n".join(doc.page_content for doc in checklist_docs)
    return parse_onboarding_steps(text)


class OnboardingManager:
    BACK_COMMANDS = {"назад", "back", "prev", "предыдущий"}
    NEXT_COMMANDS = {"далее", "next", "продолжить", "ок", "готово"}
    STATUS_COMMANDS = {"статус", "status", "где я"}

    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or settings
        self.steps = load_onboarding_steps(self.cfg)
        self.sessions: dict[str, OnboardingSession] = {}
        self.llm = self._create_llm()
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", ONBOARDING_SYSTEM_PROMPT),
                (
                    "human",
                    "Текущий шаг {step_number}/{total_steps}: {step_title}\n"
                    "Содержание шага:\n{step_body}\n\n"
                    "Сообщение сотрудника: {message}\n\n"
                    "Кратко помоги по текущему шагу и подскажи, что делать дальше.",
                ),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _create_llm(self):
        if self.cfg.llm_provider == "openai" and self.cfg.llm_api_base_url:
            base_url = self.cfg.llm_api_base_url.strip().rstrip("/")
            return ChatOpenAI(
                model=self.cfg.llm_model,
                base_url=base_url,
                api_key=self.cfg.llm_api_key or "sk-placeholder",
                temperature=0.1,
                max_tokens=1024,
            )
        return ChatOllama(
            model=self.cfg.llm_model,
            base_url=self.cfg.ollama_base_url,
            temperature=0.2,
            num_ctx=4096,
            keep_alive="24h",
        )

    def start_session(self, session_id: str | None = None) -> OnboardingSession:
        session_id = session_id or str(uuid4())
        session = OnboardingSession(session_id=session_id, steps=self.steps)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> OnboardingSession:
        if session_id not in self.sessions:
            raise KeyError(f"Сессия не найдена: {session_id}")
        return self.sessions[session_id]

    def _append_history(self, session: OnboardingSession, role: str, content: str) -> None:
        session.history.append({"role": role, "content": content})

    def handle_message(self, session_id: str, message: str) -> dict:
        session = self.get_session(session_id)
        text = message.strip().lower()

        if text in self.STATUS_COMMANDS:
            payload = session.as_dict()
            payload["reply"] = (
                f"Вы на шаге {session.current_step.number} из {session.total_steps}: "
                f"{session.current_step.title}"
            )
            return payload

        if text in self.BACK_COMMANDS:
            if session.current_index > 0:
                session.current_index -= 1
            step = session.current_step
            reply = f"Возвращаемся к шагу {step.number}: {step.title}\n\n{step.body}"
            self._append_history(session, "user", message)
            self._append_history(session, "assistant", reply)
            return {**session.as_dict(), "reply": reply}

        if text in self.NEXT_COMMANDS:
            reply = self._advance(session)
            self._append_history(session, "user", message)
            self._append_history(session, "assistant", reply)
            return {**session.as_dict(), "reply": reply}

        step = session.current_step
        reply = self.chain.invoke(
            {
                "step_number": step.number,
                "total_steps": session.total_steps,
                "step_title": step.title,
                "step_body": step.body,
                "message": message,
            }
        )
        self._append_history(session, "user", message)
        self._append_history(session, "assistant", reply)
        return {**session.as_dict(), "reply": reply}

    def update_llm_model(self, model: str) -> None:
        self.cfg.llm_model = model
        self.llm = self._create_llm()
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _advance(self, session: OnboardingSession) -> str:
        if session.current_index >= len(session.steps) - 1:
            return (
                "Адаптационный маршрут завершён. "
                "Можете задавать рабочие вопросы по документации."
            )
        session.current_index += 1
        step = session.current_step
        return f"Шаг {step.number}: {step.title}\n\n{step.body}"
