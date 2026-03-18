from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-chinese", out_dim: int = 256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0, :]
        emb = self.proj(cls)
        return {
            "embedding": emb,
            "attentions": out.attentions if output_attentions else None,
            "last_hidden_state": out.last_hidden_state,
        }


class TabEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BioNQPXAI(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        tab_in_dim: int = 13,
        text_dim: int = 256,
        tab_dim: int = 128,
        fusion_dim: int = 256,
    ):
        super().__init__()
        self.text_encoder = TextEncoder(model_name=model_name, out_dim=text_dim)
        self.tab_encoder = TabEncoder(in_dim=tab_in_dim, out_dim=tab_dim)

        self.fusion = nn.Sequential(
            nn.Linear(text_dim + tab_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.nqp_head = nn.Linear(fusion_dim, 1)
        self.fin_head = nn.Linear(fusion_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tab_x: torch.Tensor,
        output_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        text_emb = text_out["embedding"]
        tab_emb = self.tab_encoder(tab_x)

        z = torch.cat([text_emb, tab_emb], dim=-1)
        u = self.fusion(z)

        nqp_score = torch.sigmoid(self.nqp_head(u))
        fin_score = torch.sigmoid(self.fin_head(u))

        return {
            "fusion_emb": u,
            "nqp_score": nqp_score,
            "fin_score": fin_score,
            "attentions": text_out["attentions"],
            "last_hidden_state": text_out["last_hidden_state"],
            "text_emb": text_emb,
            "tab_emb": tab_emb,
        }
