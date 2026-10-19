"""已训练 Transformer 语言模型的采样与解码工具。"""

from __future__ import annotations

import argparse
import json

import torch

from .model import TransformerLM
from .tokenizer import Tokenizer


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float | None = None) -> int:
    """从最后一个位置的 logits 中采样一个 token id。

    参数：
        logits: 形状为 (vocab_size,) 的未归一化分数。
        temperature: softmax 温度；0 表示贪心解码。
        top_p: (0, 1] 内的 nucleus 采样阈值；None 表示不做过滤。
    """
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = int(torch.searchsorted(cumulative, torch.tensor(top_p, device=probs.device)).item()) + 1
        keep = sorted_indices[:cutoff]
        filtered = torch.zeros_like(probs)
        filtered[keep] = sorted_probs[:cutoff]
        probs = filtered / filtered.sum()
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_id: int | None = None,
    context_length: int | None = None,
) -> list[int]:
    """根据 prompt 生成 token id，直到遇到 EOS 或达到 max_new_tokens。"""
    device = next(model.parameters()).device
    context_length = context_length or model.context_length
    ids = list(prompt_ids)
    model.eval()
    for _ in range(max_new_tokens):
        context = torch.tensor(ids[-context_length:], dtype=torch.long, device=device).unsqueeze(0)
        logits = model(context)[0, -1]
        next_id = sample_next_token(logits, temperature=temperature, top_p=top_p)
        ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
    return ids[len(prompt_ids) :]


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: str = "cpu") -> TransformerLM:
    with open(config_path) as f:
        cfg = json.load(f)
    model = TransformerLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=cfg["rope_theta"],
        use_rope=cfg["use_rope"],
        norm=cfg["norm"],
        use_rmsnorm=cfg["use_rmsnorm"],
        ffn=cfg["ffn"],
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="从已训练的模型生成文本。")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="train.py 保存的 config.json")
    parser.add_argument("--vocab", required=True, help="tokenizer 词表文件")
    parser.add_argument("--merges", required=True, help="tokenizer merges 文件")
    parser.add_argument("--special-tokens", default='["<|endoftext|>"]')
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, json.loads(args.special_tokens))
    model = load_model_from_checkpoint(args.checkpoint, args.config, args.device)
    eos_id = None
    for token in json.loads(args.special_tokens):
        encoded = tokenizer.encode(token)
        if len(encoded) == 1:
            eos_id = encoded[0]
    prompt_ids = tokenizer.encode(args.prompt)
    new_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_id=eos_id,
    )
    print(tokenizer.decode(prompt_ids + new_ids))


if __name__ == "__main__":
    main()
