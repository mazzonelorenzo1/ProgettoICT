from pathlib import Path
from face.db import FaceDB


# Project root directory
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "face_db.pkl"


def main():
    # Initialize the database connection
    db = FaceDB(str(DB_PATH))

    # Retrieve the list of registered users
    users = db.list_users()

    print(f"Database path: {DB_PATH}\n")

    # Check if the database is empty
    if not users:
        print("No users registered.")
        return

    # Print all registered users
    print("Registered users:")
    for i, user in enumerate(users, 1):
        print(f"{i}. {user}")


if __name__ == "__main__":
    main()
