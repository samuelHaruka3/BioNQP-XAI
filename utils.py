from typing import List, Tuple

import torch
import torch.nn.functional as F


def classification_loss(out, y_nqp, y_fin):
    loss_nqp = F.binary_cross_entropy(out["nqp_score"], y_nqp.unsqueeze(1))
    loss_fin = F.binary_cross_entropy(out["fin_score"], y_fin.unsqueeze(1))
    return loss_nqp + loss_fin


def ranking_loss(scores_pos: torch.Tensor, scores_neg: torch.Tensor, margin: float = 0.2):
    return torch.relu(margin - (scores_pos - scores_neg)).mean()


def token_attention_scores(attentions, attention_mask: torch.Tensor):
    """
    attentions: tuple[num_layers] of (B, H, L, L)
    返回最后一层 CLS -> token 的平均注意力
    """
    if attentions is None:
        return None
    last = attentions[-1]            # (B, H, L, L)
    cls_to_tokens = last[:, :, 0, :] # (B, H, L)
    scores = cls_to_tokens.mean(dim=1)
    scores = scores * attention_mask
    return scores


def split_text_to_sentences(text: str) -> List[str]:
    seps = ["。", "；", ";", "\n"]
    tmp = text
    for s in seps:
        tmp = tmp.replace(s, "||")
    parts = [x.strip() for x in tmp.split("||") if x.strip()]
    return parts if parts else [text]


def aggregate_sentence_scores(
    tokenizer,
    text: str,
    token_scores: torch.Tensor,
    max_length: int = 256
) -> List[Tuple[str, float]]:
    """
    粗略把 token 分数映射回句子。
    注意：这是原型版，不是严格对齐版。
    """
    sents = split_text_to_sentences(text)
    results = []

    for sent in sents:
        enc = tokenizer(
            sent,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        attn_mask = enc["attention_mask"].squeeze(0)
        sent_len = int(attn_mask.sum().item())

        # 取前 sent_len 个 token score 做近似
        score = float(token_scores[:sent_len].mean().item()) if sent_len > 0 else 0.0
        results.append((sent, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results
