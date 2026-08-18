"""Export the sign-projected token table used by StaRSE inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from starse.loss import static_embedding_weight


def sign_project(weight: torch.Tensor) -> torch.Tensor:
    """Preserve each token norm and replace its direction with sign bits."""

    source = weight.detach().float()
    norms = torch.linalg.vector_norm(source, dim=1).clamp_min(1e-12)
    signs = torch.where(source >= 0, torch.ones_like(source), -torch.ones_like(source))
    return norms.unsqueeze(1) * signs / math.sqrt(source.shape[1])


def export_model(model_path: str, output_dir: Path) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device="cpu")
    weight = static_embedding_weight(model)
    projected = sign_project(weight)
    weight.data.copy_(projected.to(dtype=weight.dtype))

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), create_model_card=False)
    metadata = {
        "projection": "rho * sign(unit) / sqrt(dim)",
        "embedding_dim": int(projected.shape[1]),
        "vocab_size": int(projected.shape[0]),
        "storage_note": "This checkpoint stores the projected table as floats; signs and per-token norms may be packed losslessly.",
    }
    (output_dir / "projection.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Trained StaticEmbedding model")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    export_model(args.model, args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
