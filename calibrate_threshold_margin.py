import cv2
import numpy as np
from pathlib import Path
from itertools import product

from face.detector_dlib import DlibFaceDetector
from face.align_dlib import DlibAligner5pt
from face.sphereface_embedder import SphereFaceEmbedder
from face.db import FaceDB, UserRecord
from face.verifier import FaceVerifier
from face.pipeline import FacePipeline

ROOT = Path(__file__).resolve().parent.parent

PRED_PATH = ROOT / "assets" / "shape_predictor_5_face_landmarks.dat"
MODEL_PATH = ROOT / "model" / "sphere20a_20171020.pth"

CALIB_DIR = ROOT / "data" / "calib"
UNKNOWN_DIR = ROOT / "data" / "unknown"

def read_img(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError(f"Immagine non letta: {p}")
    return img

def build_temp_db(embs_by_user):
    # FaceDB "fake" in memoria: non salviamo su disco
    db = FaceDB(db_path=str(ROOT / "data" / "embeddings" / "_temp.pkl"))
    db.users = {}
    for user, embs in embs_by_user.items():
        mat = np.stack(embs, axis=0)
        template = mat.mean(axis=0)
        template = template / (np.linalg.norm(template) + 1e-12)
        db.users[user] = UserRecord(user_id=user, embeddings=embs, template=template)
    return db

def main():
    detector = DlibFaceDetector(upsample=1)
    aligner = DlibAligner5pt(str(PRED_PATH))
    embedder = SphereFaceEmbedder(str(MODEL_PATH), device="cpu")

    # 1) carica immagini utenti e calcola embeddings
    users = [d for d in CALIB_DIR.iterdir() if d.is_dir()]
    if not users:
        raise RuntimeError(f"Nessuna cartella utenti in {CALIB_DIR}")

    embs_by_user = {}
    for ud in users:
        imgs = sorted([p for p in ud.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
        if len(imgs) < 3:
            raise RuntimeError(f"Utente {ud.name}: servono almeno 3 immagini, trovate {len(imgs)}")

        embs = []
        for p in imgs[:3]:
            img = read_img(p)
            # estrazione embedding con pipeline minimale (senza verifier/db)
            rects, scores = detector.detect(img)
            if len(rects) == 0:
                raise RuntimeError(f"No face per {p}")
            best_i = int(np.argmax(np.array(scores)))
            aligned = aligner.align(img, rects[best_i])
            emb = embedder.embed(aligned)
            embs.append(emb)
        embs_by_user[ud.name] = embs

    # 2) embeddings unknown
    unknown_imgs = sorted([p for p in UNKNOWN_DIR.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    if len(unknown_imgs) == 0:
        raise RuntimeError(f"Nessuna immagine unknown in {UNKNOWN_DIR}")

    unknown_embs = []
    for p in unknown_imgs:
        img = read_img(p)
        rects, scores = detector.detect(img)
        if len(rects) == 0:
            continue
        best_i = int(np.argmax(np.array(scores)))
        aligned = aligner.align(img, rects[best_i])
        unknown_embs.append(embedder.embed(aligned))

    # 3) griglia di ricerca threshold/margin
    thresholds = np.arange(0.45, 0.86, 0.01)
    margins = np.arange(0.00, 0.21, 0.01)

    best = None  # (score, thr, mar, stats)

    # metriche: vogliamo ridurre FPR sugli unknown e mantenere TPR sui genuini
    # funzione obiettivo semplice: score = 2*TPR - 3*FPR  (pesi modificabili)
    for thr, mar in product(thresholds, margins):
        # Leave-One-Out: per ogni user, 3 split (2 enroll, 1 test)
        tp = 0
        fn = 0

        # Unknown: false positives
        fp_unknown = 0
        tn_unknown = 0

        # costruisci un verifier con questi parametri
        # NB: mode topk_mean con topk=3 va bene anche con 2 embeddings (fa mean dei disponibili)
        # Il db cambia ad ogni split, quindi il verifier lo istanziamo dopo.

        # genuini
        for user, embs in embs_by_user.items():
            for test_i in range(3):
                enroll_embs = [e for j, e in enumerate(embs) if j != test_i]
                test_emb = embs[test_i]

                # DB temporaneo con tutti gli utenti: per ciascuno usa 2 embeddings (leave-one-out per user)
                temp = {}
                for u2, e2 in embs_by_user.items():
                    if u2 == user:
                        temp[u2] = enroll_embs
                    else:
                        temp[u2] = e2  # usa tutte le 3 per gli altri (o puoi limitarle a 2)
                db = build_temp_db(temp)
                verifier = FaceVerifier(
                    db,
                    threshold=float(thr),
                    margin=float(mar),
                    mode="max",  # o "topk_mean" se è quello che userai
                    topk=3,
                    default_max_samples=4,  # 4 per tutti
                    per_user_max_samples={"Lorenzo": 8}
                )

                pred_user, best_score, second_score = verifier.identify(test_emb)
                if pred_user == user:
                    tp += 1
                else:
                    fn += 1

        # unknown
        # qui usiamo un DB “pieno” (tutte e 3 immagini per user)
        db_full = build_temp_db(embs_by_user)
        verifier_full = FaceVerifier(
            db_full,
            threshold=float(thr),
            margin=float(mar),
            mode="topk_mean",
            topk=3,
            default_max_samples=4,
            per_user_max_samples={"Lorenzo": 8}
        )

        for ue in unknown_embs:
            pred_user, best_score, second_score = verifier_full.identify(ue)
            if pred_user is None:
                tn_unknown += 1
            else:
                fp_unknown += 1

        # tassi
        TPR = tp / (tp + fn + 1e-12)          # riconoscere i veri
        FPR = fp_unknown / (fp_unknown + tn_unknown + 1e-12)  # riconoscere “unknown” come qualcuno (male)

        objective = 2.0 * TPR - 3.0 * FPR

        stats = dict(TPR=TPR, FPR=FPR, tp=tp, fn=fn, fp_unknown=fp_unknown, tn_unknown=tn_unknown)
        if best is None or objective > best[0]:
            best = (objective, float(thr), float(mar), stats)

    obj, thr_best, mar_best, stats = best
    print("BEST")
    print("threshold:", thr_best, "margin:", mar_best)
    print("stats:", stats)
    print("objective:", obj)

if __name__ == "__main__":
    main()
