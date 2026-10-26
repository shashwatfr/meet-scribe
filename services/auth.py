"""Login: bcrypt password hashing and checking. Passwords are never stored, logged or printed.

verify_login() is called by app.py and process_cli.py; hash_password() by seed_users.py.
ensure_demo_users() is called once by app.py to create the hosted demo users from secrets.
"""
import bcrypt

from config import get_value
from services import store

# Checked when the username is unknown, so a wrong username takes as long as a wrong password.
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt())

# Hosted demo users: (username, secret holding the password, role).
DEMO_SECRETS = (("admin", "DEMO_PASSWORD_ADMIN", "admin"),
                ("alice", "DEMO_PASSWORD_ALICE", "user"),
                ("bob", "DEMO_PASSWORD_BOB", "user"))


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_login(username: str, password: str) -> dict | None:
    """Return {id, username, role} for a correct login, otherwise None.

    Unknown user and wrong password give the same result in similar time.
    """
    user = store.get_user_by_username(username.strip().lower())
    stored = user["password_hash"].encode("ascii") if user else _DUMMY_HASH
    try:
        ok = bcrypt.checkpw(password.encode("utf-8"), stored)
    except ValueError:  # bcrypt 5 rejects passwords over 72 bytes
        ok = False
    if not (user and ok):
        return None
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


def ensure_demo_users() -> list[str]:
    """Hosted demo: create admin, alice and bob from secret passwords. Returns who was created.

    Only acts when all three DEMO_PASSWORD_* values are set AND the users table is empty
    (a fresh host, or one whose temporary storage was wiped). Locally, with no such
    secrets, it does nothing and seed_users.py is used instead.
    """
    passwords = {username: get_value(secret) for username, secret, _ in DEMO_SECRETS}
    if not all(passwords.values()) or store.count_users() > 0:
        return []
    for username, _, role in DEMO_SECRETS:
        store.create_user(username, hash_password(passwords[username]), role)
    return [username for username, _, _ in DEMO_SECRETS]
