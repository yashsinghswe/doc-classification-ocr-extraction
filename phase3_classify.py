"""
Phase 3 — Document Classification (Few-Shot Embedding Search)
Uses sentence-transformers/all-MiniLM-L6-v2 to embed text,
then classifies by cosine similarity to per-type centroids.

How to use:
    classifier = FewShotClassifier()
    classifier.add_examples("invoice", [text1, text2, ...])   # 5-10 examples
    result = classifier.classify(some_text)

The embedding index is persisted to models/embedding_index.json so you don't
need to re-add examples every run.
"""

import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"   # 80 MB, runs on CPU
INDEX_PATH  = "models/embedding_index.json"
CONF_THRESH = 0.75   # documents below this go to human review (Phase 6)


class FewShotClassifier:
    def __init__(self, confidence_threshold: float = CONF_THRESH):
        self.confidence_threshold = confidence_threshold
        self.centroids: dict[str, np.ndarray] = {}
        self._model = None   # lazy-load so import is fast
        self._load_index()

    # ── Model (lazy) ──────────────────────────────────────────────────────────

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Phase 3] Loading {MODEL_NAME} …")
            self._model = SentenceTransformer(MODEL_NAME)
            print("[Phase 3] Model ready.")
        return self._model

    # ── Index persistence ─────────────────────────────────────────────────────

    def _load_index(self):
        if os.path.exists(INDEX_PATH):
            with open(INDEX_PATH, "r") as f:
                raw = json.load(f)
            self.centroids = {k: np.array(v) for k, v in raw.items()}
            print(f"[Phase 3] Loaded index: {list(self.centroids.keys())}")
        else:
            print("[Phase 3] No index found — add examples with add_examples().")

    def _save_index(self):
        os.makedirs("models", exist_ok=True)
        with open(INDEX_PATH, "w") as f:
            json.dump({k: v.tolist() for k, v in self.centroids.items()}, f)

    # ── Adding examples ───────────────────────────────────────────────────────

    def add_examples(self, doc_type: str, texts: list[str]):
        """
        Encode 5-10 example texts for a document type and store the centroid.
        Call this once per type before running the pipeline.

        Example
        -------
        classifier.add_examples("invoice", [
            open("data/few_shot/invoice_1.txt").read(),
            open("data/few_shot/invoice_2.txt").read(),
        ])
        """
        if not texts:
            raise ValueError("Provide at least one example text.")

        embeddings = self.model.encode(texts, show_progress_bar=False)
        centroid = np.mean(embeddings, axis=0)
        self.centroids[doc_type] = centroid
        self._save_index()
        print(f"[Phase 3] Added '{doc_type}' ({len(texts)} examples) → index saved.")

    # ── Classification ────────────────────────────────────────────────────────

    def classify(self, text: str) -> dict:
        """
        Classify a document by cosine similarity to stored centroids.

        Returns
        -------
        {
            "predicted_type":  str,           # best match or "uncertain"
            "confidence":      float,          # 0-1
            "needs_review":    bool,
            "all_scores":      {type: score},
        }
        """
        if not self.centroids:
            return {
                "predicted_type": "unknown",
                "confidence": 0.0,
                "needs_review": True,
                "all_scores": {},
            }

        # Truncate to first 512 tokens worth of text (model max)
        truncated = text[:2000]
        embedding  = self.model.encode([truncated], show_progress_bar=False)

        scores: dict[str, float] = {}
        for doc_type, centroid in self.centroids.items():
            sim = cosine_similarity(embedding, centroid.reshape(1, -1))[0][0]
            scores[doc_type] = round(float(sim), 4)

        best_type  = max(scores, key=scores.get)
        best_score = scores[best_type]
        uncertain  = best_score < self.confidence_threshold

        return {
            "predicted_type": best_type if not uncertain else "uncertain",
            "confidence": best_score,
            "needs_review": uncertain,
            "all_scores": scores,
        }

    # ── Feedback from human review (Phase 6 calls this) ──────────────────────

    def add_from_review(self, doc_type: str, text: str):
        """
        Incrementally update the centroid when a human confirms a document type.
        This is the self-improving feedback loop described in the POC guide.
        """
        new_emb = self.model.encode([text[:2000]], show_progress_bar=False)[0]

        if doc_type in self.centroids:
            # Simple running average — good enough for POC
            self.centroids[doc_type] = (self.centroids[doc_type] + new_emb) / 2.0
        else:
            self.centroids[doc_type] = new_emb

        self._save_index()
        print(f"[Phase 3] Centroid updated for '{doc_type}' from human feedback.")

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_types(self) -> list[str]:
        return list(self.centroids.keys())

    def remove_type(self, doc_type: str):
        if doc_type in self.centroids:
            del self.centroids[doc_type]
            self._save_index()
            print(f"[Phase 3] Removed '{doc_type}' from index.")


# ── Seed helper — call once to bootstrap the index from text files ────────────

def seed_index_from_folder(folder: str = "data/few_shot"):
    """
    Expects folder structure:
        data/few_shot/
            invoice/
                example1.txt
                example2.txt
            contract/
                example1.txt
            ...

    Reads all .txt files per sub-folder and calls add_examples().
    """
    classifier = FewShotClassifier()
    base = os.path.expanduser(folder)

    if not os.path.exists(base):
        print(f"[Phase 3] Folder not found: {base}")
        return

    for doc_type in os.listdir(base):
        type_dir = os.path.join(base, doc_type)
        if not os.path.isdir(type_dir):
            continue

        texts = []
        for fname in os.listdir(type_dir):
            if fname.endswith(".txt"):
                with open(os.path.join(type_dir, fname), "r", encoding="utf-8") as f:
                    texts.append(f.read())

        if texts:
            classifier.add_examples(doc_type, texts)

    print(f"\n[Phase 3] Index seeded with types: {classifier.list_types()}")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    clf = FewShotClassifier()

    if "--seed" in sys.argv:
        seed_index_from_folder()
        sys.exit(0)

    if "--add" in sys.argv:
        # python phase3_classify.py --add invoice path/to/example.txt
        idx = sys.argv.index("--add")
        doc_type = sys.argv[idx + 1]
        text_path = sys.argv[idx + 2]
        with open(text_path) as f:
            clf.add_examples(doc_type, [f.read()])
        sys.exit(0)

    if "--classify" in sys.argv:
        idx = sys.argv.index("--classify")
        text_path = sys.argv[idx + 1]
        with open(text_path) as f:
            text = f.read()
        result = clf.classify(text)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    print("Usage:")
    print("  python phase3_classify.py --seed                        # seed index from data/few_shot/")
    print("  python phase3_classify.py --add <type> <file.txt>       # add one example")
    print("  python phase3_classify.py --classify <file.txt>         # classify a document")
