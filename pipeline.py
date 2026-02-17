import cv2


class FacePipeline:
    def __init__(self, detector, aligner, embedder, verifier, db, notifier=None):
        # Initialize pipeline with all necessary components (Dependency Injection)
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.verifier = verifier
        self.db = db
        self.notifier = notifier

    def extract_embedding_from_image(self, bgr_img):
        # Step 1: Detect faces in the image
        rects, scores = self.detector.detect(bgr_img)
        if len(rects) == 0:
            return None, {"error": "no_face"}

        import numpy as np
        # Select the face with the highest detection score
        best_i = int(np.argmax(np.array(scores)))
        rect = rects[best_i]

        # Step 2: Align the face using landmarks
        aligned = self.aligner.align(bgr_img, rect)

        # Save debug image (optional)
        cv2.imwrite("debug_aligned.jpg", aligned)

        # Step 3: Generate embedding vector
        emb = self.embedder.embed(aligned)
        return emb, {"face_score": float(scores[best_i])}

    def verify(self, bgr_img, claimed_user_id: str):
        # 1:1 Verification Process
        emb, meta = self.extract_embedding_from_image(bgr_img)

        # Handle case where no face is found
        if emb is None:
            # Send notification if notifier is configured
            if self.notifier:
                self.notifier.send(f" Verification failed: no face detected (claimed={claimed_user_id}).")
            return {"ok": False, **meta}

        # Perform verification against the claimed ID
        ok, score = self.verifier.verify_claim(emb, claimed_user_id)

        # Notify only on failure (Access Denied)
        if (not ok) and self.notifier:
            self.notifier.send(
                f"Access denied. claimed={claimed_user_id} score={float(score):.3f} "
                f"(thr={float(self.verifier.threshold):.2f})"
            )

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
        # 1:N Identification Process
        emb, meta = self.extract_embedding_from_image(bgr_img)

        if emb is None:
            if self.notifier:
                self.notifier.send("Identification failed: no face detected")
            return {"ok": False, "user": None, **meta}

        # Find the best match in the database
        user, best, second_user, second_score = self.verifier.identify(emb)

        # Notify only if the user is NOT recognized (Unknown or Ambiguous)
        if user is None and self.notifier:
            self.notifier.send(
                f"Unknown person. best={float(best):.3f} "
                f"second={float(second_score):.3f} (2nd={second_user}) "
                f"(thr={float(self.verifier.threshold):.2f}, margin={float(self.verifier.margin):.2f})"
            )

        return {
            "ok": user is not None,
            "user": user,
            "score": float(best),
            "second_user": second_user,
            "second_score": float(second_score),
            "threshold": float(self.verifier.threshold),
            "margin": float(self.verifier.margin),
            "mode": self.verifier.mode,
            "topk": self.verifier.topk,
            **meta
        }

    def enroll_user(self, user_id: str, bgr_imgs: list):
        # Extract embeddings for all provided images
        embs = []
        for img in bgr_imgs:
            emb, meta = self.extract_embedding_from_image(img)
            if emb is not None:
                embs.append(emb)

        # Ensure at least one valid face was found
        if len(embs) == 0:
            return {"ok": False, "error": "no_valid_faces_for_enroll"}

        # Update or Insert user into the database
        self.db.upsert_user(user_id, embs)
        return {"ok": True, "user": user_id, "n_samples": len(embs)}

    def delete_user(self, user_id: str):
        # Remove user from database
        if user_id not in self.db.users:
            return {"ok": False, "error": "user_not_found", "user": user_id}
        self.db.delete_user(user_id)
        return {"ok": True, "deleted_user": user_id, "users": self.db.list_users()}

    def list_users(self):
        # Retrieve list of all registered users
        users = self.db.list_users()
        return {
            "ok": True,
            "n_users": len(users),
            "users": users
        }
