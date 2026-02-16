# face/list_users.py
from pathlib import Path
from face.db import FaceDB


# ROOT del progetto (una cartella sopra /face)
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "face_db.pkl"


def main():
    db = FaceDB(str(DB_PATH))

    users = db.list_users()

    print(f"Database path: {DB_PATH}\n")

    if not users:
        print("No users registered.")
        return

    print("Registered users:")
    for i, user in enumerate(users, 1):
        print(f"{i}. {user}")


if __name__ == "__main__":
    main()
