from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path

USERS_FILE = Path("data/users.json")
_tokens: dict[str, str] = {}


def _load_users() -> dict:
    if not USERS_FILE.exists():
        return {"admin": _make_admin()}
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"admin": _make_admin()}


def _save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def _make_admin() -> dict:
    return {
        "password": _hash_password("admin"),
        "role": "admin",
        "created_at": "",
    }


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{h}"


def _verify_password(password: str, hashed: str) -> bool:
    if ":" not in hashed:
        return False
    salt, h = hashed.split(":", 1)
    return h == hashlib.sha256((salt + password).encode()).hexdigest()


def login(username: str, password: str) -> str | None:
    users = _load_users()
    user = users.get(username)
    if not user or not _verify_password(password, user["password"]):
        return None
    token = secrets.token_hex(32)
    _tokens[token] = username
    return token


def validate_token(token: str) -> dict | None:
    username = _tokens.get(token)
    if not username:
        return None
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    return {"username": username, "role": user.get("role", "user")}


def register(username: str, password: str, role: str = "user") -> dict:
    users = _load_users()
    if username in users:
        raise ValueError("Пользователь уже существует")
    users[username] = {
        "password": _hash_password(password),
        "role": role,
        "created_at": "",
    }
    _save_users(users)
    return {"username": username, "role": role}


def list_users(token: str) -> list[dict]:
    info = validate_token(token)
    if not info or info["role"] != "admin":
        raise PermissionError("Требуются права администратора")
    users = _load_users()
    return [
        {"username": k, "role": v.get("role", "user")}
        for k, v in users.items()
    ]


def delete_user(token: str, username: str) -> None:
    info = validate_token(token)
    if not info or info["role"] != "admin":
        raise PermissionError("Требуются права администратора")
    if username == "admin":
        raise ValueError("Нельзя удалить администратора")
    users = _load_users()
    if username not in users:
        raise FileNotFoundError("Пользователь не найден")
    del users[username]
    _save_users(users)


def logout(token: str) -> None:
    _tokens.pop(token, None)
