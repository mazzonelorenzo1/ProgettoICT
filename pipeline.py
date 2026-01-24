class FacePipeline:
    def __init__(self, detector, aligner, embedder, verifier, db):
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.verifier = verifier
        self.db = db

    def extract_embedding_from_image(self, bgr_img):
        rects, scores = self.detector.detect(bgr_img)
        if len(rects) == 0:
            return None, {"error": "no_face"}

        import numpy as np
        best_i = int(np.argmax(np.array(scores)))
        rect = rects[best_i]

        aligned = self.aligner.align(bgr_img, rect)
        emb = self.embedder.embed(aligned)
        return emb, {"face_score": float(scores[best_i])}

    def verify(self, bgr_img, claimed_user_id: str):
        emb, meta = self.extract_embedding_from_image(bgr_img)
        if emb is None:
            return {"ok": False, **meta}

        ok, score = self.verifier.verify_claim(emb, claimed_user_id)
        return {
            "ok": ok,
            "claimed_user": claimed_user_id,
            "score": float(score),
            "threshold": float(self.verifier.threshold),
            "mode": getattr(self.verifier, "mode", "template"),
            "topk": int(getattr(self.verifier, "topk", 1)),
            **meta
        }

    def identify(self, bgr_img):
        emb, meta = self.extract_embedding_from_image(bgr_img)
        if emb is None:
            return {"ok": False, "user": None, **meta}

        user, best, second = self.verifier.identify(emb)

        return {
            "ok": user is not None,
            "user": user,
            "score": float(best),
            "second_score": float(second),
            "threshold": float(self.verifier.threshold),
            "margin": float(getattr(self.verifier, "margin", 0.0)),
            "mode": getattr(self.verifier, "mode", "template"),
            "topk": int(getattr(self.verifier, "topk", 1)),
            **meta
        }

    def enroll_user(self, user_id: str, bgr_imgs: list):
        embs = []
        for img in bgr_imgs:
            emb, meta = self.extract_embedding_from_image(img)
            if emb is not None:
                embs.append(emb)
        if len(embs) == 0:
            return {"ok": False, "error": "no_valid_faces_for_enroll"}
        self.db.upsert_user(user_id, embs)
        return {"ok": True, "user": user_id, "n_samples": len(embs)}
