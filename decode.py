from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_lm import AdapterTransformerLM, TrainConfig, build_tokenizer


DEFAULT_WORK_DIR = Path("runs/tinystories")


def load_train_config(work_dir: Path) -> TrainConfig:
    config_path = work_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    payload.pop("vocab_size", None)
    return TrainConfig(**payload)


def resolve_checkpoint(work_dir: Path, checkpoint_name: str) -> Path:
    if checkpoint_name.endswith(".pt") or "\\" in checkpoint_name or "/" in checkpoint_name:
        checkpoint_path = Path(checkpoint_name)
    else:
        checkpoint_path = work_dir / "checkpoints" / f"{checkpoint_name}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def load_model(
    work_dir: Path,
    checkpoint_name: str,
    device: str,
    tokenizer_vocab_size: int,
) -> tuple[AdapterTransformerLM, TrainConfig]:
    cfg = load_train_config(work_dir)

    model = AdapterTransformerLM(
        vocab_size=tokenizer_vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
    ).to(device)

    checkpoint_path = resolve_checkpoint(work_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
) -> int:
    next_token_logits = logits[:, -1, :]

    if temperature <= 0:
        return int(torch.argmax(next_token_logits, dim=-1).item())

    next_token_logits = next_token_logits / temperature
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)), dim=-1)
        cutoff = values[:, -1].unsqueeze(-1)
        next_token_logits = next_token_logits.masked_fill(next_token_logits < cutoff, float("-inf"))

    probs = torch.softmax(next_token_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: AdapterTransformerLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
) -> str:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("Prompt encodes to zero tokens. Provide a non-empty prompt.")

    generated = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        input_ids = generated[:, -model.context_length :]
        logits = model(input_ids)
        next_token = sample_next_token(logits, temperature=temperature, top_k=top_k)
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        generated = torch.cat([generated, next_token_tensor], dim=1)

    return tokenizer.decode(generated[0].tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a trained checkpoint and generate text.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint name under work_dir/checkpoints without .pt, or a direct .pt path.",
    )
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device, for example cpu or cuda.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.work_dir)
    tokenizer, vocab_size = build_tokenizer(cfg)
    model, _ = load_model(args.work_dir, args.checkpoint, args.device, vocab_size)

    text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    print(text)


if __name__ == "__main__":
    main()
