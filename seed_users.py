"""Creates the three demo users (alice, bob, admin) with throwaway passwords.

Run by hand once after setup. Safe to run twice: existing users are left unchanged.

Usage: uv run python seed_users.py
"""
from services import store
from services.auth import hash_password

# DEMO ONLY: throwaway passwords for a local demo, listed in the README.
# Never reuse them anywhere, and never deploy this app with these accounts.
DEMO_USERS = [
    ("alice", "alice-demo-2026", "user"),
    ("bob", "bob-demo-2026", "user"),
    ("admin", "admin-demo-2026", "admin"),
]


def seed() -> list[tuple[str, str]]:
    """Returns [(username, 'created' | 'exists'), ...]."""
    results = []
    for username, password, role in DEMO_USERS:
        if store.get_user_by_username(username):
            results.append((username, "exists"))  # keep the existing hash, don't reset
            continue
        created = store.create_user(username, hash_password(password), role)
        results.append((username, "created" if created else "exists"))
    return results


if __name__ == "__main__":
    for username, outcome in seed():
        print(f"{username:<8} {outcome}")
