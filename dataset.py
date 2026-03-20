import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


TAB_FEATURES = [
    "revenue_growth",
    "net_profit_growth",
    "ocf_margin",
    "cash_to_revenue",
    "rd_ratio",
    "liability_ratio",
    "overseas_share",
    "capacity_ton_log",
    "utilization",
    "event_count",
    "commercial_event_count",
    "partner_count",
    "scene_count",
    "rule_score",
]

def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b not in (0, 0.0, None) else 0.0

def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_patent_text_map(patent_rows):
    patent_map = {}
    for row in patent_rows:
        firm_id = row["firm_id"]
        text = f"{row.get('title','')}。{row.get('abstract','')}。{row.get('claims','')}"
        patent_map.setdefault(firm_id, []).append(text)
    return {fid: "\n".join(texts) for fid, texts in patent_map.items()}
def build_partner_scene_features(edges_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    从简化图边中统计：
    - partner_count
    - scene_count
    """
    result: Dict[str, Dict[str, float]] = {}

    if edges_df.empty:
        return result

    firm_ids = set(edges_df.loc[edges_df["src_type"] == "firm", "src_id"].tolist())

    for firm_id in firm_ids:
        sub = edges_df[edges_df["src_id"] == firm_id]

        partner_count = sub[sub["relation"].isin(["strategic_coop_with", "invested_by", "supplies_to"])]["dst_id"].nunique()
        scene_count = sub[sub["dst_type"] == "scenario"]["dst_id"].nunique()

        result[firm_id] = {
            "partner_count": float(partner_count),
            "scene_count": float(scene_count),
        }

    return result


def preprocess_firms(firms_df: pd.DataFrame, graph_feat: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = firms_df.copy()
    # 基础派生
    df["ocf_margin"] = df.apply(lambda r: safe_div(r["ocf"], r["revenue"]), axis=1)
    df["cash_to_revenue"] = df.apply(lambda r: safe_div(r["cash"], r["revenue"]), axis=1)
    df["rd_ratio"] = df.apply(lambda r: safe_div(r["rd_expense"], r["revenue"]), axis=1)
    df["liability_ratio"] = df.apply(lambda r: safe_div(r["total_liabilities"], r["total_assets"]), axis=1)
    df["overseas_share"] = df.apply(lambda r: safe_div(r["overseas_revenue"], r["revenue"]), axis=1)
    df["capacity_ton_log"] = np.log1p(df["capacity_ton"].clip(lower=0))

    # 如果没有增速列，可先置零
    for col in ["revenue_growth", "net_profit_growth", "event_count", "commercial_event_count", "partner_count", "scene_count"]:
        if col not in df.columns:
            df[col] = 0.0

    # 合并图统计特征
    def get_graph_value(firm_id: str, key: str) -> float:
        return float(graph_feat.get(firm_id, {}).get(key, 0.0))

    df["partner_count"] = df["firm_id"].apply(lambda x: get_graph_value(x, "partner_count"))
    df["scene_count"] = df["firm_id"].apply(lambda x: get_graph_value(x, "scene_count"))

    for col in TAB_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0).astype(float)

    return df

class FirmDataset(Dataset):
    def __init__(
        self,
        firms_csv: str,
        announcements_jsonl: str,
        patents_jsonl:str,
        graph_edges_csv: str,
        model_name: str = "bert-base-chinese",
        max_length: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        self.firms_df = pd.read_csv(firms_csv)
        self.ann_rows = load_jsonl(announcements_jsonl)
        self.edges_df = pd.read_csv(graph_edges_csv) if Path(graph_edges_csv).exists() else pd.DataFrame()

        graph_feat = build_partner_scene_features(self.edges_df)
        self.firms_df = preprocess_firms(self.firms_df, graph_feat)

        ann_map: Dict[str, List[str]] = {}
        for row in self.ann_rows:
            firm_id = row["firm_id"]
            text = f"{row.get('title', '')}。{row.get('text', '')}"
            ann_map.setdefault(firm_id, []).append(text)

        self.text_map: Dict[str, str] = {
            fid: "\n".join(texts) for fid, texts in ann_map.items()
        }

        self.rows = self.firms_df.to_dict("records")
        self.patent_rows = load_jsonl(patents_jsonl)
        self.patent_text_map = build_patent_text_map(self.patent_rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        firm_id = row["firm_id"]
        text = self.text_map.get(firm_id, "无公告文本。")
        patent_text = self.patent_text_map.get(firm_id, "无专利文本。")

        enc = self.tokenizer(
            text,
            patent_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        pat_enc = self.tokenizer(
            patent_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        tab_x = np.array([row[col] for col in TAB_FEATURES], dtype=np.float32)

        y_nqp = float(row.get("label_nqp", row.get("tr1_label_nqp", 0)))
        y_fin = float(row.get("label_fin", row.get("tr1_label_fin", 0)))

        return {
            "firm_id": firm_id,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "patent_input_ids": pat_enc["input_ids"].squeeze(0),
            "patent_attention_mask": pat_enc["attention_mask"].squeeze(0),
            "raw_patent_text": patent_text,
            "tab_x": torch.tensor(tab_x, dtype=torch.float32),
            "y_nqp": torch.tensor(y_nqp, dtype=torch.float32),
            "y_fin": torch.tensor(y_fin, dtype=torch.float32),
            "raw_text": text,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "firm_id": [x["firm_id"] for x in batch],
        "raw_text": [x["raw_text"] for x in batch],
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "tab_x": torch.stack([x["tab_x"] for x in batch], dim=0),
        "patent_input_ids": torch.stack([x["patent_input_ids"] for x in batch], dim=0),
        "patent_attention_mask": torch.stack([x["patent_attention_mask"] for x in batch], dim=0),
        "raw_patent_text": [x["raw_patent_text"] for x in batch],
        "y_nqp": torch.stack([x["y_nqp"] for x in batch], dim=0),
        "y_fin": torch.stack([x["y_fin"] for x in batch], dim=0),
    }



from torch.utils.data import DataLoader


def quick_test_dataset():
    # 1. 初始化数据集（指向 data 目录）
    print("🔍 初始化数据集...")
    dataset = FirmDataset(
        firms_csv="./data/firms.csv",
        announcements_jsonl="./data/announcements.jsonl",
        graph_edges_csv="./data/graph_edges.csv",
        model_name="bert-base-chinese",
        max_length=256
    )

    # 核心验证1：基础信息
    print(f"\n✅ 核心验证 - 基础信息")
    print(f"   样本总数: {len(dataset)}")
    print(f"   表格特征数: {len(dataset[0]['tab_x'])} (预期13个)")

    # 核心验证2：单样本关键特征
    sample = dataset[1]
    print(f"\n✅ 核心验证 - 单样本特征")
    print(f"   企业ID: {sample['firm_id']}")
    print(f"   文本token数: {sample['input_ids'].shape[0]} (预期256)")
    print(f"   财务特征示例(前3个): {sample['tab_x'][:3].numpy().round(4)}")
    print(f"   预测标签(nqp/fin): {sample['y_nqp'].item():.4f} / {sample['y_fin'].item():.4f}")

    # 核心验证3：批量加载
    print(f"\n✅ 核心验证 - 批量加载")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    print(f"   批量input_ids形状: {batch['input_ids'].shape} (预期[2,256])")
    print(f"   批量表格特征形状: {batch['tab_x'].shape} (预期[2,13])")
    print(f"   批量标签形状: {batch['y_nqp'].shape} (预期[2])")

    print("\n🎉 所有核心功能验证完成！")


if __name__ == "__main__":
    # 解决中文显示问题（可选）
    import pandas as pd

    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    quick_test_dataset()