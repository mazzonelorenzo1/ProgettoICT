        # 1:N Identification - Compare embedding against ALL users
        if not self.db.users:
            return None, -1.0, None, -1.0

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
            # Filter out very low scores to speed up sorting
            if s > -0.5:
                scored.append((uid, float(s)))

        if not scored:
            return None, -1.0, None, -1.0

        # Sort candidates by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        best_user, best_score = scored[0]
        if len(scored) > 1:
            second_user, second_score = scored[1]
        else:
            second_user, second_score = None, -1.0

        # Acceptance Criteria:
        # 1. Score must exceed Threshold
        # 2. Difference between 1st and 2nd best must exceed Margin
        ok = (best_score >= self.threshold) and ((best_score - second_score) >= self.margin)

        return (
            best_user if ok else None,
            best_score,
            second_user,
            second_score
        )
