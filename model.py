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

from torch_geometric.nn import HGTConv
import torch.nn.functional as F

class GraphEncoderHGT(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=128, num_heads=4, num_layers=2, metadata=None):
        super().__init__()
        self.metadata = metadata
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                HGTConv(
                    in_channels=-1,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads
                )
            )

        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict, target_index=None, target_node_type="firm"):
        h_dict = x_dict
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}

        firm_x = h_dict[target_node_type]   # [num_firm_nodes, hidden_dim]

        if target_index is not None:
            firm_x = firm_x[target_index]   # 只取当前 batch 对应 firm

        return self.proj(firm_x)

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
        tab_in_dim: int = 14,
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

class BioNQPXAIPlus(nn.Module):
    def __init__(
        self,
        model_name="bert-base-chinese",
        tab_in_dim=14,
        ann_dim=256,
        patent_dim=256,
        tab_dim=128,
        fusion_dim=256,
    ):
        super().__init__()
        self.ann_encoder = TextEncoder(model_name=model_name, out_dim=ann_dim)
        self.pat_encoder = TextEncoder(model_name=model_name, out_dim=patent_dim)
        self.tab_encoder = TabEncoder(in_dim=tab_in_dim, out_dim=tab_dim)

        # 先不上 graph，所以这里只拼 ann + patent + tab
        self.fusion = nn.Sequential(
            nn.Linear(ann_dim + patent_dim + tab_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.nqp_head = nn.Linear(fusion_dim, 1)
        self.fin_head = nn.Linear(fusion_dim, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        patent_input_ids,
        patent_attention_mask,
        tab_x,
        output_attentions=False,
    ):
        ann_out = self.ann_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        pat_out = self.pat_encoder(
            input_ids=patent_input_ids,
            attention_mask=patent_attention_mask,
            output_attentions=output_attentions,
        )

        tab_emb = self.tab_encoder(tab_x)

        z = torch.cat(
            [ann_out["embedding"], pat_out["embedding"], tab_emb],
            dim=-1
        )
        u = self.fusion(z)

        return {
            "fusion_emb": u,
            "nqp_score": torch.sigmoid(self.nqp_head(u)),
            "fin_score": torch.sigmoid(self.fin_head(u)),
            "ann_attentions": ann_out["attentions"],
            "pat_attentions": pat_out["attentions"],
        }