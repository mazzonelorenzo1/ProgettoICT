from pathlib import Path
from face.db import FaceDB

# Define project root directory
ROOT = Path(__file__).resolve().parent.parent

def main():
    import argparse
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id")
    parser.add_argument("--db", default=str(ROOT / "face_db.pkl"))
    args = parser.parse_args()

    # Initialize the database
    db = FaceDB(args.db)

    # Check if user exists before deleting
    before = db.list_users()
    if args.user_id not in before:
        print(f"[WARN] User '{args.user_id}' not found. Existing users: {before}")
        return

    # Delete the user
    db.delete_user(args.user_id)
    after = db.list_users()

    # Print success message and current users
    print(f"[OK] Deleted '{args.user_id}'.")
    print("Users now:", after)


if __name__ == "__main__":
    main()
