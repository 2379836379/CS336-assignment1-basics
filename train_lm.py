from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from tests.adapters import (
    get_adamw_cls,
    get_tokenizer,
    run_cross_entropy,
    run_get_batch,
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    run_load_checkpoint,
    run_save_checkpoint,
    run_train_bpe,
    run_transformer_lm,
)
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode


DEFAULT_TRAIN_PATH = Path("data/TinyStoriesV2-GPT4-train.txt")
DEFAULT_VALID_PATH = Path("data/TinyStoriesV2-GPT4-valid.txt")
DEFAULT_WORK_DIR = Path("runs/tinystories")


@dataclass
class TrainConfig:
    train_path: str
    valid_path: str
    work_dir: str
    tokenizer_backend: str
    bpe_vocab_size: int
    bpe_special_tokens: list[str]
    context_length: int
    batch_size: int
    steps: int
    eval_every: int
    eval_batches: int
    save_every: int
    log_every: int
    lr: float
    min_lr: float
    warmup_iters: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str
    seed: int
    max_train_lines: int | None
    max_valid_lines: int | None
    rebuild_cache: bool
    resume: bool


class AttentionWeights(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj_weight = nn.Parameter(torch.empty(d_model, d_model))
        self.output_proj_weight = nn.Parameter(torch.empty(d_model, d_model))

    def reset_parameters(self, std: float) -> None:
        for param in (
            self.q_proj_weight,
            self.k_proj_weight,
            self.v_proj_weight,
            self.output_proj_weight,
        ):
            nn.init.normal_(param, mean=0.0, std=std)


class FeedForwardWeights(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1_weight = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2_weight = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3_weight = nn.Parameter(torch.empty(d_ff, d_model))

    def reset_parameters(self, std: float) -> None:
        for param in (self.w1_weight, self.w2_weight, self.w3_weight):
            nn.init.normal_(param, mean=0.0, std=std)


class TransformerLayerWeights(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.attn = AttentionWeights(d_model)
        self.ln1_weight = nn.Parameter(torch.ones(d_model))
        self.ffn = FeedForwardWeights(d_model, d_ff)
        self.ln2_weight = nn.Parameter(torch.ones(d_model))

    def reset_parameters(self, std: float) -> None:
        self.attn.reset_parameters(std)
        self.ffn.reset_parameters(std)
        nn.init.ones_(self.ln1_weight)
        nn.init.ones_(self.ln2_weight)


class AdapterTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings_weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self.layers = nn.ModuleList(
            [TransformerLayerWeights(d_model=d_model, d_ff=d_ff) for _ in range(num_layers)]
        )
        self.ln_final_weight = nn.Parameter(torch.ones(d_model))
        self.lm_head_weight = nn.Parameter(torch.empty(vocab_size, d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 0.02
        nn.init.normal_(self.token_embeddings_weight, mean=0.0, std=std)
        nn.init.normal_(self.lm_head_weight, mean=0.0, std=std)
        nn.init.ones_(self.ln_final_weight)
        for layer in self.layers:
            layer.reset_parameters(std)

    def _weights(self) -> dict[str, torch.Tensor]:
        weights: dict[str, torch.Tensor] = {
            "token_embeddings.weight": self.token_embeddings_weight,
            "ln_final.weight": self.ln_final_weight,
            "lm_head.weight": self.lm_head_weight,
        }
        for idx, layer in enumerate(self.layers):
            prefix = f"layers.{idx}."
            weights[f"{prefix}attn.q_proj.weight"] = layer.attn.q_proj_weight
            weights[f"{prefix}attn.k_proj.weight"] = layer.attn.k_proj_weight
            weights[f"{prefix}attn.v_proj.weight"] = layer.attn.v_proj_weight
            weights[f"{prefix}attn.output_proj.weight"] = layer.attn.output_proj_weight
            weights[f"{prefix}ln1.weight"] = layer.ln1_weight
            weights[f"{prefix}ffn.w1.weight"] = layer.ffn.w1_weight
            weights[f"{prefix}ffn.w2.weight"] = layer.ffn.w2_weight
            weights[f"{prefix}ffn.w3.weight"] = layer.ffn.w3_weight
            weights[f"{prefix}ln2.weight"] = layer.ln2_weight
        return weights

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return run_transformer_lm(
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            weights=self._weights(),
            in_indices=token_ids,
        )


def load_fixture_gpt2_tokenizer() -> tuple[Any, int]:
    vocab_path = FIXTURES_PATH / "gpt2_vocab.json"
    merges_path = FIXTURES_PATH / "gpt2_merges.txt"
    byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    with vocab_path.open(encoding="utf-8") as f:
        gpt2_vocab = json.load(f)

    vocab = {
        token_id: bytes([byte_decoder[ch] for ch in token_text])
        for token_text, token_id in gpt2_vocab.items()
    }

    merges: list[tuple[bytes, bytes]] = []
    with merges_path.open(encoding="utf-8") as f:
        for line in f:
            cleaned = line.rstrip()
            parts = cleaned.split(" ")
            if cleaned and len(parts) == 2:
                merges.append(
                    (
                        bytes([byte_decoder[ch] for ch in parts[0]]),
                        bytes([byte_decoder[ch] for ch in parts[1]]),
                    )
                )

    tokenizer = get_tokenizer(vocab, merges, ["<|endoftext|>"])
    return tokenizer, len(vocab)


def build_tokenizer(cfg: TrainConfig) -> tuple[Any, int]:
    if cfg.tokenizer_backend == "fixture-gpt2":
        return load_fixture_gpt2_tokenizer()

    if cfg.tokenizer_backend == "train-bpe":
        tokenizer_path = Path(cfg.work_dir) / "tokenizer.pt"
        if tokenizer_path.exists():
            state = torch.load(tokenizer_path, weights_only=False)
            vocab = state["vocab"]
            merges = state["merges"]
            special_tokens = state["special_tokens"]
        else:
            print(f"Training BPE tokenizer on {cfg.train_path}")
            vocab, merges = run_train_bpe(
                input_path=cfg.train_path,
                vocab_size=cfg.bpe_vocab_size,
                special_tokens=cfg.bpe_special_tokens,
            )
            special_tokens = cfg.bpe_special_tokens
            tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "vocab": vocab,
                    "merges": merges,
                    "special_tokens": special_tokens,
                },
                tokenizer_path,
            )
        tokenizer = get_tokenizer(vocab, merges, special_tokens)
        return tokenizer, len(vocab)

    raise ValueError(f"Unsupported tokenizer backend: {cfg.tokenizer_backend}")


def limited_lines(path: Path, max_lines: int | None) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        if max_lines is None:
            yield from f
        else:
            yield from itertools.islice(f, max_lines)


def cache_name(split: str, cfg: TrainConfig) -> str:
    line_limit = "all" if (cfg.max_train_lines if split == "train" else cfg.max_valid_lines) is None else str(
        cfg.max_train_lines if split == "train" else cfg.max_valid_lines
    )
    return f"{split}.{cfg.tokenizer_backend}.{line_limit}.uint32.bin"


def open_token_cache(path: Path) -> np.memmap:
    item_size = np.dtype(np.uint32).itemsize
    num_items = path.stat().st_size // item_size
    if num_items <= 1:
        raise ValueError(f"Token cache has too few tokens: {path}")
    return np.memmap(path, dtype=np.uint32, mode="r", shape=(num_items,))


def write_token_cache(
    tokenizer: Any,
    src_path: Path,
    out_path: Path,
    max_lines: int | None,
) -> np.memmap:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buffer: list[int] = []
    total_tokens = 0
    flush_size = 1 << 16

    print(f"Encoding {src_path} -> {out_path}")
    with out_path.open("wb") as out_file:
        for token_id in tokenizer.encode_iterable(limited_lines(src_path, max_lines)):
            buffer.append(token_id)
            if len(buffer) >= flush_size:
                np.asarray(buffer, dtype=np.uint32).tofile(out_file)
                total_tokens += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=np.uint32).tofile(out_file)
            total_tokens += len(buffer)

    print(f"Wrote {total_tokens:,} tokens.")
    return open_token_cache(out_path)


def maybe_build_cache(
    tokenizer: Any,
    split: str,
    src_path: Path,
    max_lines: int | None,
    cfg: TrainConfig,
) -> np.memmap:
    out_path = Path(cfg.work_dir) / "cache" / cache_name(split, cfg)
    if cfg.rebuild_cache or not out_path.exists():
        return write_token_cache(tokenizer, src_path, out_path, max_lines)
    print(f"Using cached tokens: {out_path}")
    return open_token_cache(out_path)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer_cls = get_adamw_cls()
    return optimizer_cls(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )


def current_lr(step: int, cfg: TrainConfig) -> float:
    if cfg.warmup_iters > 0:
        return run_get_lr_cosine_schedule(
            it=step,
            max_learning_rate=cfg.lr,
            min_learning_rate=cfg.min_lr,
            warmup_iters=cfg.warmup_iters,
            cosine_cycle_iters=cfg.steps,
        )

    progress = min(max(step / max(cfg.steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * float(cosine)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        x, y = run_get_batch(dataset, batch_size=batch_size, context_length=context_length, device=device)
        logits = model(x)
        loss = run_cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def save_config(cfg: TrainConfig, vocab_size: int) -> None:
    config_path = Path(cfg.work_dir) / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    payload["vocab_size"] = vocab_size
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_path = Path(cfg.train_path)
    valid_path = Path(cfg.valid_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation file not found: {valid_path}")

    tokenizer, vocab_size = build_tokenizer(cfg)
    save_config(cfg, vocab_size)

    train_tokens = maybe_build_cache(tokenizer, "train", train_path, cfg.max_train_lines, cfg)
    valid_tokens = maybe_build_cache(tokenizer, "valid", valid_path, cfg.max_valid_lines, cfg)

    model = AdapterTransformerLM(
        vocab_size=vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
    ).to(cfg.device)
    optimizer = build_optimizer(model, cfg)

    checkpoint_dir = Path(cfg.work_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    latest_path = checkpoint_dir / "latest.pt"
    if cfg.resume and latest_path.exists():
        print(f"Resuming from {latest_path}")
        start_step = run_load_checkpoint(latest_path, model, optimizer) + 1

    print(
        f"Training on {cfg.device}: vocab={vocab_size}, "
        f"train_tokens={len(train_tokens):,}, valid_tokens={len(valid_tokens):,}"
    )

    best_valid_loss = float("inf")
    model.train()

    for step in range(start_step, cfg.steps):
        lr = current_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = run_get_batch(
            train_tokens,
            batch_size=cfg.batch_size,
            context_length=cfg.context_length,
            device=cfg.device,
        )
        logits = model(x)
        loss = run_cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        run_gradient_clipping(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            print(f"step={step:06d} train_loss={loss.item():.4f} lr={lr:.6e}")

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            train_loss = estimate_loss(
                model=model,
                dataset=train_tokens,
                batch_size=cfg.batch_size,
                context_length=cfg.context_length,
                device=cfg.device,
                eval_batches=cfg.eval_batches,
            )
            valid_loss = estimate_loss(
                model=model,
                dataset=valid_tokens,
                batch_size=cfg.batch_size,
                context_length=cfg.context_length,
                device=cfg.device,
                eval_batches=cfg.eval_batches,
            )
            print(
                f"eval step={step:06d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}"
            )
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                run_save_checkpoint(model, optimizer, step, checkpoint_dir / "best.pt")

        if step % cfg.save_every == 0 or step == cfg.steps - 1:
            run_save_checkpoint(model, optimizer, step, latest_path)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a small Transformer LM while reusing implementations from tests/adapters.py."
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--tokenizer-backend",
        choices=["fixture-gpt2", "train-bpe"],
        default="fixture-gpt2",
    )
    parser.add_argument("--bpe-vocab-size", type=int, default=8000)
    parser.add_argument(
        "--bpe-special-token",
        action="append",
        default=["<|endoftext|>"],
        help="Repeat to add multiple special tokens when using --tokenizer-backend train-bpe.",
    )
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu or cuda.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-train-lines", type=int, default=None)
    parser.add_argument("--max-valid-lines", type=int, default=5000)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    return TrainConfig(
        train_path=str(args.train_path),
        valid_path=str(args.valid_path),
        work_dir=str(args.work_dir),
        tokenizer_backend=args.tokenizer_backend,
        bpe_vocab_size=args.bpe_vocab_size,
        bpe_special_tokens=list(dict.fromkeys(args.bpe_special_token)),
        context_length=args.context_length,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        save_every=args.save_every,
        log_every=args.log_every,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
        seed=args.seed,
        max_train_lines=args.max_train_lines,
        max_valid_lines=args.max_valid_lines,
        rebuild_cache=args.rebuild_cache,
        resume=args.resume,
    )


if __name__ == "__main__":
    train(parse_args())
