import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ai.faiss_index import FaissIndexStore
from backend.app.core.config import get_settings
from backend.app.core.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a saved FAISS reference index by embedding offset.")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    metadata_path = Path(args.metadata_path) if args.metadata_path else resolve_project_path(settings.metadata_dir) / "latest_reference.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embeddings = np.load(metadata["embeddings_path"])
    if embeddings.shape[0] == 0:
        raise SystemExit("Reference library has no embeddings to query.")
    if args.query_offset < 0 or args.query_offset >= embeddings.shape[0]:
        raise SystemExit(f"query-offset must be between 0 and {embeddings.shape[0] - 1}.")

    store = FaissIndexStore(metric=metadata.get("faiss_metric", "cosine"))
    index = store.load(Path(metadata["faiss_index_path"]))
    result = store.search(index, embeddings[args.query_offset : args.query_offset + 1], k=args.k)

    print(
        json.dumps(
            {
                "reference_version": metadata["reference_version"],
                "query_offset": args.query_offset,
                "neighbors": [
                    {"embedding_offset": int(index_value), "score": float(score)}
                    for index_value, score in zip(result.indices[0], result.scores[0])
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
