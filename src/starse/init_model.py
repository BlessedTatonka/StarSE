"""Create the 512-dimensional initialization used by StaRSE."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


def build_initial_matrix(
    matrix: np.ndarray,
    *,
    target_dim: int = 512,
    sif_a: float = 5e-5,
    zipf_exponent: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the PCA + SIF/Zipf recipe used for the released model."""

    vocab_size, hidden_dim = matrix.shape
    if target_dim > hidden_dim:
        raise ValueError(f"target_dim={target_dim} exceeds source dimension {hidden_dim}")

    components_to_remove = math.ceil(target_dim / 100)
    n_components = min(hidden_dim, target_dim + components_to_remove)
    components_to_remove = max(0, n_components - target_dim)
    pca = PCA(n_components=n_components, svd_solver="auto", whiten=False, random_state=42)
    reduced = pca.fit_transform(matrix)
    if components_to_remove:
        reduced = reduced[:, components_to_remove:]
    reduced = reduced[:, :target_dim]

    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    zipf = 1.0 / np.power(ranks, zipf_exponent)
    probabilities = zipf / zipf.sum()
    weights = sif_a / (sif_a + probabilities)
    initialized = np.ascontiguousarray((reduced * weights[:, None]).astype(np.float32))

    metadata = {
        "source": "AutoModel.get_input_embeddings().weight",
        "source_shape": [int(vocab_size), int(hidden_dim)],
        "final_shape": list(initialized.shape),
        "pca_components": int(n_components),
        "leading_components_removed": int(components_to_remove),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "sif_a": float(sif_a),
        "zipf_exponent": float(zipf_exponent),
    }
    return initialized, metadata


def initialize_model(
    *,
    base_model: str,
    output_dir: Path,
    target_dim: int,
    sif_a: float,
    zipf_exponent: float,
    trust_remote_code: bool,
) -> None:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import StaticEmbedding

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    source = model.get_input_embeddings().weight.detach().to(dtype=torch.float32, device="cpu").numpy()
    matrix, metadata = build_initial_matrix(
        source,
        target_dim=target_dim,
        sif_a=sif_a,
        zipf_exponent=zipf_exponent,
    )
    del model

    static_model = SentenceTransformer(modules=[StaticEmbedding(tokenizer=tokenizer, embedding_weights=matrix)])
    output_dir.mkdir(parents=True, exist_ok=True)
    static_model.save_pretrained(str(output_dir), create_model_card=False)
    metadata.update({"base_model": base_model, "target_dim": target_dim})
    (output_dir / "initialization.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="ai-forever/ruBert-base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-dim", type=int, default=512)
    parser.add_argument("--sif-a", type=float, default=5e-5)
    parser.add_argument("--zipf-exponent", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    initialize_model(
        base_model=args.base_model,
        output_dir=args.output_dir.resolve(),
        target_dim=args.target_dim,
        sif_a=args.sif_a,
        zipf_exponent=args.zipf_exponent,
        trust_remote_code=args.trust_remote_code,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
