import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from glob import glob
from sklearn.metrics import confusion_matrix, accuracy_score

# Import classes
from face.detector_dlib import DlibFaceDetector
from face.align_dlib import DlibAligner5pt
from face.sphereface_embedder import SphereFaceEmbedder

# Fixed threshold to validate
TEST_THRESHOLD = 0.28


def get_all_embeddings_with_names(root_dir, detector, aligner, embedder):
    """
    Extracts embeddings and also keeps track of filenames
    for the error report.
    """
    embeddings = []
    labels = []
    filenames = []  # New list for filenames

    # Search for jpg and png
    image_paths = glob(os.path.join(root_dir, "*", "*.jpg")) + \
                  glob(os.path.join(root_dir, "*", "*.png"))

    print(f" Loading and analyzing {len(image_paths)} images...")

    for path in image_paths:
        # Label = folder name (ex. "Alessandro")
        label = os.path.basename(os.path.dirname(path))
        # Filename = file name (ex. "Ale_1.jpg")
        fname = os.path.basename(path)

        img = cv2.imread(path)
        if img is None: continue

        rects, scores = detector.detect(img)
        if len(rects) == 0: continue

        best_i = np.argmax(scores)
        rect = rects[best_i]

        aligned = aligner.align(img, rect)
        emb = embedder.embed(aligned)

        embeddings.append(emb)
        labels.append(label)
        filenames.append(f"{label}/{fname}")

    return np.array(embeddings), np.array(labels), np.array(filenames)


def run_validation(embeddings, labels, filenames, threshold):
    # 1. L2 Normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)

    # 2. Similarity Matrix
    sim_matrix = np.dot(embeddings, embeddings.T)

    y_true = []
    y_pred = []

    # Lists to save specific errors
    false_positives = []  # (Imposters passed)
    false_negatives = []  # (Genuine blocked)

    n = len(embeddings)
    print(f"  Starting Cross-Validation on {n} images ({n * (n - 1) // 2} unique comparisons)...")

    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j]

            is_same_person = (labels[i] == labels[j])
            prediction = (score >= threshold)

            y_true.append(1 if is_same_person else 0)
            y_pred.append(1 if prediction else 0)

            # Error capturing
            if prediction and not is_same_person:
                # System said YES, but they are DIFFERENT -> False Positive (SEVERE)
                false_positives.append({
                    "img1": filenames[i],
                    "img2": filenames[j],
                    "score": score
                })

            elif not prediction and is_same_person:
                # System said NO, but they are SAME -> False Negative (Annoying)
                false_negatives.append({
                    "img1": filenames[i],
                    "img2": filenames[j],
                    "score": score
                })

    # Metrics calculation
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print("\n" + "=" * 70)
    print(f"  DETAILED VALIDATION REPORT (Threshold = {threshold})")
    print("=" * 70)
    print(f"GENERAL METRICS:")
    print(f"    Accuracy:       {accuracy * 100:.2f}%")
    print(f"    FAR (Security): {far * 100:.2f}%  (False Positives: {fp})")
    print(f"    FRR (Usability):{frr * 100:.2f}%  (False Negatives: {fn})")
    print("-" * 70)

    # Print specific errors
    print("\n  FALSE POSITIVES DETAIL (Security Holes):")
    if len(false_positives) == 0:
        print("     No errors, system is 100% secure on this dataset.")
    else:
        # Sort by descending score (most severe first)
        false_positives.sort(key=lambda x: x['score'], reverse=True)
        for err in false_positives:
            print(f"       WRONG MATCH: {err['img1']} <--> {err['img2']}")
            print(f"       Score: {err['score']:.4f} (Threshold {threshold})")

    print("\n FALSE NEGATIVES DETAIL (Missed Recognitions):")
    if len(false_negatives) == 0:
        print("   No errors, system always recognizes registered users.")
    else:
        # Sort by ascending score (furthest first)
        false_negatives.sort(key=lambda x: x['score'])
        for err in false_negatives:
            print(f"   WRONG REJECTION: {err['img1']} <--> {err['img2']}")
            print(f"   Score: {err['score']:.4f} (Threshold {threshold})")

    print("=" * 70)

    # Plot
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Pred: Diff', 'Pred: Same'],
                yticklabels=['Real: Diff', 'Real: Same'])
    plt.title(f'Error Matrix (Thr={threshold})')
    plt.tight_layout()
    plt.savefig("validation_errors.png")
    print("Plot saved to 'validation_errors.png'")


def main():
    detector = DlibFaceDetector()
    aligner = DlibAligner5pt("../assets/shape_predictor_5_face_landmarks.dat")
    embedder = SphereFaceEmbedder("../model/sphere20a_20171020.pth")
    DATA_PATH = "../data/calib"

    if os.path.exists(DATA_PATH):
        emb, lbl, fnames = get_all_embeddings_with_names(DATA_PATH, detector, aligner, embedder)
        run_validation(emb, lbl, fnames, threshold=TEST_THRESHOLD)
    else:
        print("Error: Data folder not found.")


if __name__ == "__main__":
    main()
