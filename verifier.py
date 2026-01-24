# src/face/verifier.py
import numpy as np
from typing import Tuple, Optional, List, Dict


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def _limited_embeddings(rec, max_samples: int) -> List[np.ndarray]:
    embs = getattr(rec, "embeddings", None)
    if not embs:
        return []
    # Usa gli ultimi N (in genere sono i più recenti/aggiunti per ultimi)
    return [e for e in embs[-max_samples:] if e is not None]


def _score_against_user(
    emb: np.ndarray,
    rec,
    mode: str,
    k: int,
    max_samples_for_user: int
) -> float:
    """
    mode:
      - "max": massimo tra i campioni (best)
      - "topk_mean": media dei top-K score (più stabile)
      - "template": usa solo rec.template
    """
    if mode != "template":
        embs = _limited_embeddings(rec, max_samples_for_user)
        if embs:
            scores = [cosine_sim(emb, e) for e in embs]
            scores.sort(reverse=True)
            if mode == "max":
                return scores[0]
            kk = min(k, len(scores))
            return float(np.mean(scores[:kk]))

    tpl = getattr(rec, "template", None)
    if tpl is None:
        return -1.0
    return cosine_sim(emb, tpl)


class FaceVerifier:
    def __init__(
        self,
        face_db,
        threshold: float = 0.65,
        margin: float = 0.10,
        mode: str = "topk_mean",              # <-- qui "best" = max
        topk: int = 3,                  # usato solo se mode="topk_mean"
        default_max_samples: int = 4,   # <-- 4 per tutti
        per_user_max_samples: Dict[str, int] = None  # <-- override, es. Lorenzo:8
    ):
        self.db = face_db
        self.threshold = float(threshold)
        self.margin = float(margin)
        self.mode = mode
        self.topk = int(topk)
        self.default_max_samples = int(default_max_samples)
        self.per_user_max_samples = per_user_max_samples or {}

    def _max_samples(self, user_id: str) -> int:
        return int(self.per_user_max_samples.get(user_id, self.default_max_samples))

    def verify_claim(self, emb: np.ndarray, claimed_user_id: str) -> Tuple[bool, float]:
        rec = self.db.users.get(claimed_user_id)
        if rec is None:
            return False, -1.0

        score = _score_against_user(
            emb, rec,
            mode=self.mode,
            k=self.topk,
            max_samples_for_user=self._max_samples(claimed_user_id)
        )
        return (score >= self.threshold), float(score)

    def identify(self, emb: np.ndarray) -> Tuple[Optional[str], float, float]:
        if not self.db.users:
            return None, -1.0, -1.0

        scored = []
        for uid, rec in self.db.users.items():
            if rec is None:
                continue
            s = _score_against_user(
                emb, rec,
                mode=self.mode,
                k=self.topk,
                max_samples_for_user=self._max_samples(uid)
            )
            if s > -0.5:
                scored.append((uid, float(s)))

        if not scored:
            return None, -1.0, -1.0

        scored.sort(key=lambda x: x[1], reverse=True)
        best_user, best_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else -1.0

        ok = (best_score >= self.threshold) and ((best_score - second_score) >= self.margin)
        return (best_user if ok else None), float(best_score), float(second_score)

