from dataclasses import dataclass
import numpy as np
import os, pickle
from typing import Dict, List

@dataclass
class UserRecord:
    # Data structure to hold user information
    user_id: str
    embeddings: List[np.ndarray]  # List of raw embedding vectors (512,)
    template: np.ndarray          # Normalized mean vector (centroid)

class FaceDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        # Dictionary to store user records in memory
        self.users: Dict[str, UserRecord] = {}
        self.load()

    def load(self):
        # Load database from disk if it exists
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.users = pickle.load(f)

    def save(self):
        # Ensure directory exists and save database to disk
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.users, f)

    def upsert_user(self, user_id: str, new_embeddings: List[np.ndarray]):
        all_emb = []
        # If user exists, retrieve existing embeddings
        if user_id in self.users:
            all_emb.extend(self.users[user_id].embeddings)
        
        # Add new embeddings
        all_emb.extend(new_embeddings)

        # Compute new template (Centroid)
        mat = np.stack(all_emb, axis=0)
        template = mat.mean(axis=0)
        # Normalize template for Cosine Similarity
        template = template / (np.linalg.norm(template) + 1e-12)

        # Update user record and save to disk
        self.users[user_id] = UserRecord(user_id=user_id, embeddings=all_emb, template=template)
        self.save()

    def delete_user(self, user_id: str):
        # Remove user and persist changes
        if user_id in self.users:
            del self.users[user_id]
            self.save()

    def list_users(self):
        # Return a sorted list of registered user IDs
        return sorted(self.users.keys())
