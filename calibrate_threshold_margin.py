import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

# Import classes
from face.detector_dlib import DlibFaceDetector
from face.align_dlib import DlibAligner5pt
from face.sphereface_embedder import SphereFaceEmbedder


def get_all_embeddings(root_dir, detector, aligner, embedder):
    """
    Loads all the images in the directories, computes the embeddings
    and gives back a list of (embedding, label name).
    """
    embeddings = []
    labels = []
    paths = []

    # Searches all the images recursively
    image_paths = glob(os.path.join(root_dir, "*", "*.jpg")) + \
                  glob(os.path.join(root_dir, "*", "*.png")) + \
                  glob(os.path.join(root_dir, "*", "*.jpeg"))

    print(f"{len(image_paths)} images found. Computation ongoing...")

    for path in tqdm(image_paths):
        # Extracts the name of the label directory (ex. "Alessandro")
        label = os.path.basename(os.path.dirname(path))

        img = cv2.imread(path)
        if img is None: continue

        # 1. Detect
        rects, scores = detector.detect(img)
        if len(rects) == 0:
            print(f"⚠️ Nessun volto in: {path}")
            continue

        # Takes the face with the higher score
        best_i = np.argmax(scores)
        rect = rects[best_i]

        # 2. Align
        aligned = aligner.align(img, rect)
        aligned = aligner.align(img, rect)
        # Saves for debugging
        cv2.imwrite(f"debug_calib_{label}_{os.path.basename(path)}", aligned)

        # 3. Embed
        emb = embedder.embed(aligned)

        embeddings.append(emb)
        labels.append(label)
        paths.append(path)

    return np.array(embeddings), np.array(labels)


def evaluate_thresholds(embeddings, labels):
    """
    Computes the cosine similarity and finds the best threshold
    """
    n = len(embeddings)
    if n < 2:
        print("Too few dat to calibrate")
        return

    print(f"Compute the similarity on {n} faces ({n * n} confrontations)...")

    # Normalize the vectors to length 1 (L2 Norm)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-12)

    # Cosine similarity
    sim_matrix = np.dot(embeddings_norm, embeddings_norm.T)

    pos_scores = []  # Same person
    neg_scores = []  # Different person

    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle
            score = sim_matrix[i, j]

            # Debug strange case: if the score is absurd we truncate it
            if score > 1.0: score = 1.0
            if score < -1.0: score = -1.0

            if labels[i] == labels[j]:
                pos_scores.append(score)
            else:
                neg_scores.append(score)

    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    print(f"Genuine couples (same person) {len(pos_scores)}")
    print(f"Impostor couples (different person) {len(neg_scores)}")

    if len(pos_scores) == 0:
        print("No same person couples, add more photos")
        return

    # Research of the best threshold
    # We search between -1 and 1 (cosine similarity range)
    thresholds = np.arange(-0.2, 0.8, 0.01)
    best_acc = 0
    best_thr = 0

    for thr in thresholds:
        tp = np.sum(pos_scores >= thr)
        tn = np.sum(neg_scores < thr)
        # Balanced accuracy to not favore the ones with a lot of negatives
        tpr = tp / len(pos_scores) if len(pos_scores) > 0 else 0
        tnr = tn / len(neg_scores) if len(neg_scores) > 0 else 0
        acc = (tpr + tnr) / 2  # Balanced Accuracy

        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    # Metrics computation
    mean_pos = np.mean(pos_scores)
    std_pos = np.std(pos_scores)
    mean_neg = np.mean(neg_scores)
    std_neg = np.std(neg_scores)


    print("\n" + "=" * 40)
    print(f"🏆 Calibration results (Cosine Similarity):")
    print("=" * 40)
    print(f"Best Threshold: {best_thr:.3f}")
    print(f"Accuracy (Bal): {best_acc * 100:.2f}%")
    print("-" * 20)
    print(f"Average score same person: {mean_pos:.3f} (±{std_pos:.3f})")
    print(f"Average score different person: {mean_neg:.3f} (±{std_neg:.3f})")
    print("=" * 40)

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.hist(pos_scores, bins=50, alpha=0.6, color='green', label='Stessa Persona', range=(-0.5, 1.0))
    plt.hist(neg_scores, bins=50, alpha=0.6, color='red', label='Impostori', range=(-0.5, 1.0))
    plt.axvline(best_thr, color='blue', linestyle='--', label=f'Soglia ({best_thr:.2f})')
    plt.axvline(mean_pos, color='green', linestyle=':', label='Media Pos')
    plt.title('Cosine similarity distribution (normalized)')
    plt.xlabel('Cosine Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("calibration_plot.png")
    print("Plot saved as 'calibration_plot.png'")


def main():
    # Components setup
    print("Models initialization...")
    detector = DlibFaceDetector()
    aligner = DlibAligner5pt("../assets/shape_predictor_5_face_landmarks.dat")
    embedder = SphereFaceEmbedder("../model/sphere20a_20171020.pth")
    DATA_PATH = "../data/calib"

    if not os.path.exists(DATA_PATH):
        print(f"Crea una cartella {DATA_PATH} con sottocartelle per persona e foto dentro.")
        return

    emb, lbl = get_all_embeddings(DATA_PATH, detector, aligner, embedder)
    evaluate_thresholds(emb, lbl)


if __name__ == "__main__":
    main()
