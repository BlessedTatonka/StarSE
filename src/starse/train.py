"""Train StaRSE on a source-balanced collection of Parquet sentence pairs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

from starse.loss import SignProjectedContrastiveLoss


class CyclingRoundRobinBatchSampler:
    """Repeat smaller sources so every source contributes one batch per round."""

    def __init__(
        self,
        *,
        dataset: Any,
        batch_samplers: list[Any],
        dataset_names: Sequence[str],
        primary_dataset: str,
    ) -> None:
        if len(dataset.datasets) != len(batch_samplers):
            raise ValueError("dataset and batch sampler counts differ")
        if primary_dataset not in dataset_names:
            raise ValueError(f"Unknown primary dataset {primary_dataset!r}")
        if any(len(sampler) == 0 for sampler in batch_samplers):
            raise ValueError("Every training source must contain at least one complete batch")
        self.dataset = dataset
        self.batch_samplers = batch_samplers
        self.dataset_names = list(dataset_names)
        self.primary_index = self.dataset_names.index(primary_dataset)
        self.batch_size = int(getattr(batch_samplers[0], "batch_size", 1))
        self.drop_last = bool(getattr(batch_samplers[0], "drop_last", True))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.batch_samplers[self.primary_index]) * len(self.batch_samplers)

    def _iterator(self, source_index: int, cycle: int) -> Iterator[list[int]]:
        sampler = self.batch_samplers[source_index]
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch * 100_000 + cycle)
        return iter(sampler)

    def __iter__(self) -> Iterator[list[int]]:
        lengths = [len(source) for source in self.dataset.datasets]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)

        iterators = [self._iterator(index, 0) for index in range(len(self.batch_samplers))]
        cycles = [0] * len(self.batch_samplers)
        rounds = len(self.batch_samplers[self.primary_index])
        for _ in range(rounds):
            for source_index in range(len(self.batch_samplers)):
                try:
                    batch = next(iterators[source_index])
                except StopIteration:
                    cycles[source_index] += 1
                    iterators[source_index] = self._iterator(source_index, cycles[source_index])
                    batch = next(iterators[source_index])
                yield [index + offsets[source_index] for index in batch]


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Training config must be a YAML object")
    for key in ("run_name", "primary_dataset", "sources", "training"):
        if key not in payload:
            raise ValueError(f"Missing required config key: {key}")
    if not isinstance(payload["sources"], list) or not payload["sources"]:
        raise ValueError("sources must be a non-empty list")
    return payload


def resolve_files(pattern: str, data_root: Path) -> list[Path]:
    candidate = Path(pattern).expanduser()
    if not candidate.is_absolute():
        candidate = data_root / candidate
    if candidate.is_dir():
        files = sorted(candidate.glob("*.parquet"))
    elif any(character in str(candidate) for character in "*?[]"):
        files = [Path(item) for item in sorted(glob.glob(str(candidate)))]
    else:
        files = [candidate]
    files = [path.resolve() for path in files if path.is_file()]
    if not files:
        raise FileNotFoundError(f"No Parquet files matched {pattern!r} below {data_root}")
    return files


def load_pair_dataset(spec: dict[str, Any], data_root: Path) -> Any:
    from datasets import Dataset

    name = str(spec["name"])
    anchor_column = str(spec.get("anchor_column", "anchor"))
    positive_column = str(spec.get("positive_column", "positive"))
    files = resolve_files(str(spec["parquet"]), data_root)
    dataset = Dataset.from_parquet([str(path) for path in files], columns=[anchor_column, positive_column])
    max_rows = int(spec.get("max_rows", 0) or 0)
    if max_rows > 0 and len(dataset) > max_rows:
        dataset = dataset.select(range(max_rows))
    if anchor_column != "anchor":
        dataset = dataset.rename_column(anchor_column, "anchor")
    if positive_column != "positive":
        dataset = dataset.rename_column(positive_column, "positive")
    dataset = dataset.select_columns(["anchor", "positive"])
    if len(dataset) == 0:
        raise ValueError(f"Dataset {name!r} is empty")
    return dataset


def load_eval_dataset(pattern: str, data_root: Path) -> Any:
    return load_pair_dataset(
        {"name": "validation", "parquet": pattern, "anchor_column": "anchor", "positive_column": "positive"},
        data_root,
    )


def train(
    *,
    config: dict[str, Any],
    data_root: Path,
    model_path: str,
    output_dir: Path,
    eval_parquet: str | None,
    max_steps_override: int | None,
    batch_size_override: int | None,
    disable_bf16: bool,
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesSymmetricRankingLoss
    from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
    from sentence_transformers.sentence_transformer.training_args import (
        BatchSamplers,
        MultiDatasetBatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    train_datasets = {str(spec["name"]): load_pair_dataset(spec, data_root) for spec in config["sources"]}
    primary_dataset = str(config["primary_dataset"])
    if primary_dataset not in train_datasets:
        raise ValueError(f"primary_dataset={primary_dataset!r} is not present in sources")

    eval_pattern = eval_parquet if eval_parquet is not None else config.get("eval_parquet")
    eval_dataset = load_eval_dataset(str(eval_pattern), data_root) if eval_pattern else None
    model = SentenceTransformer(model_path)
    objective = str(config.get("objective", "contrastive"))
    loss_settings = dict(config.get("loss") or {})
    if objective == "contrastive":
        loss = MultipleNegativesSymmetricRankingLoss(
            model=model,
            scale=float(loss_settings.get("scale", 20.0)),
        )
    elif objective == "sign_projected":
        teacher = SentenceTransformer(model_path)
        loss = SignProjectedContrastiveLoss(
            model=model,
            teacher=teacher,
            scale=float(loss_settings["scale"]),
            pair_weight=float(loss_settings["pair_weight"]),
            teacher_kl_weight=float(loss_settings["teacher_kl_weight"]),
            cube_weight=float(loss_settings.get("cube_weight", 0.0)),
            balance_weight=float(loss_settings.get("balance_weight", 0.0)),
            independence_weight=float(loss_settings.get("independence_weight", 0.0)),
            unit_prior_weight=float(loss_settings.get("unit_prior_weight", 0.0)),
            regularizer_sample_size=int(loss_settings.get("regularizer_sample_size", 8192)),
        )
    else:
        raise ValueError(f"Unknown training objective: {objective!r}")
    settings = dict(config["training"])
    max_steps = int(max_steps_override or settings["max_steps"])
    batch_size = int(batch_size_override or settings["per_device_train_batch_size"])
    bf16 = bool(settings.get("bf16", True)) and not disable_bf16
    scheduler_type = str(settings["lr_scheduler_type"])
    scheduler_kwargs = dict(settings.get("lr_scheduler_kwargs") or {})

    class StaRSETrainer(SentenceTransformerTrainer):
        def get_multi_dataset_batch_sampler(
            self,
            dataset: Any,
            batch_samplers: list[Any],
            generator: Any | None = None,
            seed: int | None = 0,
        ) -> Any:
            del generator, seed
            return CyclingRoundRobinBatchSampler(
                dataset=dataset,
                batch_samplers=batch_samplers,
                dataset_names=list(self.train_dataset.keys()),
                primary_dataset=primary_dataset,
            )

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        run_name=str(config["run_name"]),
        max_steps=max_steps,
        num_train_epochs=1,
        learning_rate=float(settings["learning_rate"]),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(settings.get("per_device_eval_batch_size", 256)),
        gradient_accumulation_steps=int(settings.get("gradient_accumulation_steps", 1)),
        warmup_steps=int(settings["warmup_steps"]),
        weight_decay=float(settings["weight_decay"]),
        lr_scheduler_type=scheduler_type,
        lr_scheduler_kwargs=scheduler_kwargs,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(settings.get("eval_steps", 100)),
        save_strategy="steps",
        save_steps=int(settings.get("save_steps", 1000)),
        save_total_limit=None,
        logging_steps=int(settings.get("logging_steps", 1)),
        report_to="none",
        bf16=bf16,
        bf16_full_eval=bf16,
        dataloader_drop_last=bool(settings.get("dataloader_drop_last", False)),
        dataloader_num_workers=int(settings.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=True,
        load_best_model_at_end=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        seed=int(config.get("seed", 20260509)),
        data_seed=int(config.get("seed", 20260509)),
        remove_unused_columns=False,
        accelerator_config={"dispatch_batches": False},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = {
        "run_name": config["run_name"],
        "objective": objective,
        "primary_dataset": primary_dataset,
        "sources": [
            {
                "name": spec["name"],
                "parquet": spec["parquet"],
                "anchor_column": spec.get("anchor_column", "anchor"),
                "positive_column": spec.get("positive_column", "positive"),
                "max_rows": spec.get("max_rows"),
            }
            for spec in config["sources"]
        ],
        "loss": loss_settings,
        "training": {**settings, "max_steps": max_steps, "per_device_train_batch_size": batch_size, "bf16": bf16},
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(public_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    trainer = StaRSETrainer(
        model=model,
        args=args,
        train_dataset=train_datasets,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    result = trainer.train()
    trainer.save_model()
    metrics = dict(result.metrics)
    (output_dir / "train_result.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Initialization model or previous-stage checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-parquet", default=None, help="Override the config's eval Parquet path")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config.resolve())
    if args.print_config:
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        return 0
    metrics = train(
        config=config,
        data_root=args.data_root.resolve(),
        model_path=args.model,
        output_dir=args.output_dir.resolve(),
        eval_parquet=args.eval_parquet,
        max_steps_override=args.max_steps,
        batch_size_override=args.batch_size,
        disable_bf16=args.no_bf16,
    )
    print(json.dumps({"status": "ok", "metrics": metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
