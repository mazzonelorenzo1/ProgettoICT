import numpy as np
import os, pickle
from typing import Dict, List

@dataclass
class UserRecord:
    user_id: str
    embeddings: List[np.ndarray]  # lista di (512,)
    template: np.ndarray          # media normalizzata

class FaceDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.users: Dict[str, UserRecord] = {}
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.users = pickle.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.users, f)

    def upsert_user(self, user_id: str, new_embeddings: List[np.ndarray]):
        all_emb = []
        if user_id in self.users:
            all_emb.extend(self.users[user_id].embeddings)
        all_emb.extend(new_embeddings)

        mat = np.stack(all_emb, axis=0)
        template = mat.mean(axis=0)
        template = template / (np.linalg.norm(template) + 1e-12)

        self.users[user_id] = UserRecord(user_id=user_id, embeddings=all_emb, template=template)
        self.save()

    def delete_user(self, user_id: str):
        if user_id in self.users:
            del self.users[user_id]
            self.save()

    def list_users(self):
        return sorted(self.users.keys())
