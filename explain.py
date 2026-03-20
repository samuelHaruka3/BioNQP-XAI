import json
from pathlib import Path

import networkx as nx
import pandas as pd
import shap
import torch

from dataset import FirmDataset, TAB_FEATURES
from model import BioNQPXAI, BioNQPXAIPlus
from utils import token_attention_scores, aggregate_sentence_scores

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
def build_graph(edges_csv: str) -> nx.DiGraph:
    G = nx.DiGraph()
    edges_df = pd.read_csv(edges_csv)

    for _, r in edges_df.iterrows():
        src = r["src_id"]
        dst = r["dst_id"]
        rel = r["relation"]
        weight = float(r.get("weight", 1.0))
        G.add_edge(src, dst, relation=rel, weight=weight)

    return G


def top_k_paths(G: nx.DiGraph, source: str, k: int = 3, cutoff: int = 4):
    # 默认找去 scenario / project / capital 的路径
    targets = [n for n in G.nodes if n != source]
    results = []

    for tgt in targets:
        try:
            for path in nx.all_simple_paths(G, source=source, target=tgt, cutoff=cutoff):
                score = 0.0
                for u, v in zip(path[:-1], path[1:]):
                    score += G[u][v].get("weight", 1.0)
                results.append((path, score))
        except nx.NetworkXNoPath:
            continue

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]


def explain_text(model, dataset, firm_id: str, device):
    idx = None
    for i, row in enumerate(dataset.rows):
        if row["firm_id"] == firm_id:
            idx = i
            break

    if idx is None:
        raise ValueError(f"Firm {firm_id} not found")

    item = dataset[idx]
    input_ids = item["input_ids"].unsqueeze(0).to(device)
    attention_mask = item["attention_mask"].unsqueeze(0).to(device)
    patent_input_ids = item["patent_input_ids"].unsqueeze(0).to(device)
    patent_attention_mask = item["patent_attention_mask"].unsqueeze(0).to(device)
    tab_x = item["tab_x"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            patent_input_ids=patent_input_ids,
            patent_attention_mask=patent_attention_mask,
            tab_x=tab_x,
            output_attentions=True,
        )

    # 公告 attention
    ann_scores = token_attention_scores(out["ann_attentions"], attention_mask)[0].cpu()
    ann_sent_scores = aggregate_sentence_scores(
        dataset.tokenizer,
        item["raw_text"],
        ann_scores,
        max_length=dataset.max_length,
    )

    # 专利 attention
    pat_scores = token_attention_scores(out["pat_attentions"], patent_attention_mask)[0].cpu()
    pat_sent_scores = aggregate_sentence_scores(
        dataset.tokenizer,
        item["raw_patent_text"],
        pat_scores,
        max_length=dataset.max_length,
    )

    print(f"\n=== Text explanation for {firm_id} ===")
    print(f"NQP score: {float(out['nqp_score'].item()):.4f}")
    print(f"FIN score: {float(out['fin_score'].item()):.4f}")

    print("\n--- Top announcement sentences ---")
    for sent, score in ann_sent_scores[:5]:
        print(f"[{score:.4f}] {sent}")

    print("\n--- Top patent sentences ---")
    for sent, score in pat_sent_scores[:5]:
        print(f"[{score:.4f}] {sent}")
def explain_paths(edges_csv: str, firm_id: str):
    G = build_graph(edges_csv)
    paths = top_k_paths(G, source=firm_id, k=5)

    print(f"\n=== Graph paths for {firm_id} ===")
    for path, score in paths:
        print(f"[{score:.2f}] {' -> '.join(path)}")


def explain_shap(model, dataset, firm_id: str, device):
    """
    v2 简化版：
    固定公告 embedding + 固定专利 embedding，仅解释 tab_x 对 nqp_score 的贡献。
    """
    idx = None
    for i, row in enumerate(dataset.rows):
        if row["firm_id"] == firm_id:
            idx = i
            break

    if idx is None:
        raise ValueError(f"Firm {firm_id} not found")

    item = dataset[idx]
    model.eval()

    with torch.no_grad():
        ann_out = model.ann_encoder(
            input_ids=item["input_ids"].unsqueeze(0).to(device),
            attention_mask=item["attention_mask"].unsqueeze(0).to(device),
            output_attentions=False,
        )
        pat_out = model.pat_encoder(
            input_ids=item["patent_input_ids"].unsqueeze(0).to(device),
            attention_mask=item["patent_attention_mask"].unsqueeze(0).to(device),
            output_attentions=False,
        )

        fixed_ann_emb = ann_out["embedding"].detach()
        fixed_pat_emb = pat_out["embedding"].detach()

    import numpy as np

    background_rows = []
    for j in range(min(len(dataset), 20)):
        background_rows.append(dataset[j]["tab_x"].numpy())

    background_np = np.array(background_rows, dtype=np.float32)
    background = torch.tensor(background_np, dtype=torch.float32)

    def model_fn(x_numpy):
        x = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        with torch.no_grad():
            ann_rep = fixed_ann_emb.repeat(x.shape[0], 1)
            pat_rep = fixed_pat_emb.repeat(x.shape[0], 1)
            tab_emb = model.tab_encoder(x)

            z = torch.cat([ann_rep, pat_rep, tab_emb], dim=-1)
            u = model.fusion(z)
            y = torch.sigmoid(model.nqp_head(u))

        return y.cpu().numpy().reshape(-1)

    background_np = background.numpy()
    test_x = item["tab_x"].unsqueeze(0).numpy()

    explainer = shap.KernelExplainer(model_fn, background_np)
    shap_values = explainer.shap_values(test_x)

    values = shap_values[0] if isinstance(shap_values, list) else shap_values
    values = values[0]
    base = float(explainer.expected_value if not isinstance(explainer.expected_value, list)
                 else explainer.expected_value[0])

    pairs = list(zip(TAB_FEATURES, values, test_x[0]))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n=== SHAP explanation for {firm_id} ===")
    print(f"Base value: {base:.4f}")
    for name, shap_v, raw_v in pairs[:10]:
        print(f"{name:24s} raw={raw_v:.4f} shap={shap_v:.4f}")
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FirmDataset(
        firms_csv="data/firms.csv",
        announcements_jsonl="data/announcements.jsonl",
        graph_edges_csv="data/graph_edges.csv",
        patents_jsonl="data/patents.jsonl",
        model_name="bert-base-chinese",
        max_length=256,
    )

    model = BioNQPXAIPlus(
        model_name="bert-base-chinese",
        tab_in_dim=len(TAB_FEATURES),
    ).to(device)

    ckpt = Path("checkpoint.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        print("checkpoint.pt not found, using random weights")

    firm_id = "F001"

    explain_text(model, dataset, firm_id, device)
    explain_paths("data/graph_edges.csv", firm_id)
    explain_shap(model, dataset, firm_id, device)


if __name__ == "__main__":
    main()