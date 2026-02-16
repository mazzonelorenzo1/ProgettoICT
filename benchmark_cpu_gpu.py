import time
import numpy as np
import cv2
import torch

from face.detector_dlib import DlibFaceDetector
from face.align_dlib import DlibAligner5pt
from face.sphereface_embedder import SphereFaceEmbedder
from face.db import FaceDB
from face.verifier import FaceVerifier
from face.pipeline import FacePipeline
from pathlib import Path


def load_all_images(root_dir="../data"):
    root = Path(root_dir)
    imgs = []

    # Recursively load images from directory
    for path in root.rglob("*"):
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(path))
            if img is not None:
                imgs.append(img)

    return imgs


def sync(device: str):
    # Synchronize CUDA to ensure accurate GPU timing
    if device == "cuda":
        torch.cuda.synchronize()


def stats_ms(arr):
    # Compute timing statistics (mean, median, 95th percentile)
    arr = np.array(arr, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def build_pipeline(device: str):
    # Initialize all pipeline components
    detector = DlibFaceDetector()
    aligner = DlibAligner5pt("../assets/shape_predictor_5_face_landmarks.dat")
    embedder = SphereFaceEmbedder("../model/sphere20a_20171020.pth", device=device)

    db = FaceDB("../face_db.pkl")
    verifier = FaceVerifier(
        db,
        threshold=0.28,
        margin=0.02,
        mode="topk_mean",
        default_max_samples=4,
        per_user_max_samples={"Lorenzo": 8}
    )
    return FacePipeline(detector, aligner, embedder, verifier, db, notifier=None)


def bench_end_to_end(pipeline, imgs, device: str, warmup=5, runs=30):
    # Warmup runs to stabilize system
    for _ in range(warmup):
        _ = pipeline.identify(imgs[0])
    sync(device)

    times = []
    for i in range(runs):
        img = imgs[i % len(imgs)]
        sync(device)
        t0 = time.perf_counter()

        # Measure full identification process
        _ = pipeline.identify(img)

        sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return stats_ms(times)


def bench_embedding_only(pipeline, imgs, device: str, warmup=20, runs=200):
    # Pre-compute aligned crops to isolate embedding performance
    aligned_list = []
    for img in imgs:
        rects, scores = pipeline.detector.detect(img)
        if len(rects) == 0:
            continue
        best_i = int(np.argmax(np.array(scores)))
        aligned = pipeline.aligner.align(img, rects[best_i])
        aligned_list.append(aligned)

    if not aligned_list:
        raise RuntimeError("No faces found for embedding-only benchmark")

    # Warmup the embedder
    for _ in range(warmup):
        _ = pipeline.embedder.embed(aligned_list[0])
    sync(device)

    times = []
    for i in range(runs):
        aligned = aligned_list[i % len(aligned_list)]
        sync(device)
        t0 = time.perf_counter()

        # Measure only the embedding inference time
        _ = pipeline.embedder.embed(aligned)

        sync(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return stats_ms(times)


def main():
    # Load dataset for benchmarking
    imgs = load_all_images("../data")

    if not imgs:
        raise RuntimeError("No images loaded for benchmark")

    # Run tests on CPU and GPU (if available)
    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue

        print("\n==============================")
        print("DEVICE:", device)
        if device == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))

        pipeline = build_pipeline(device=device)

        # Execute benchmarks
        e2e = bench_end_to_end(pipeline, imgs, device=device, warmup=5, runs=30)
        emb = bench_embedding_only(pipeline, imgs, device=device, warmup=20, runs=200)

        print("End-to-end identify:", e2e)
        print("Embedding-only:", emb)


if __name__ == "__main__":
    main()B
