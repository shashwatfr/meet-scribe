"""Login and demo-user seeding. Temp database, no API calls.
Run by pytest (uv run pytest).
"""
import seed_users
from services import store
from services.auth import hash_password, verify_login


def test_login_right_wrong_and_unknown(temp_db):
    store.create_user("alice", hash_password("right-pw"), "user")
    user = verify_login("alice", "right-pw")
    assert user["username"] == "alice" and user["role"] == "user" and "password_hash" not in user
    assert verify_login(" Alice ", "right-pw") is not None  # username trimmed and lowercased
    assert verify_login("alice", "wrong-pw") is None
    assert verify_login("nobody", "right-pw") is None  # same answer as a wrong password


def test_password_over_72_bytes_is_rejected_not_crashing(temp_db):
    store.create_user("alice", hash_password("right-pw"), "user")
    assert verify_login("alice", "x" * 100) is None


def test_seed_is_idempotent(temp_db):
    first = seed_users.seed()
    hashes = {u: store.get_user_by_username(u)["password_hash"] for u, _ in first}
    second = seed_users.seed()
    assert [o for _, o in first] == ["created"] * 3
    assert [o for _, o in second] == ["exists"] * 3
    assert {u: store.get_user_by_username(u)["password_hash"] for u, _ in second} == hashes
    assert store.get_user_by_username("admin")["role"] == "admin"
    assert verify_login("alice", "alice-demo-2026") is not None
