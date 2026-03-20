# graph_builder.py
import torch
import pandas as pd
from torch_geometric.data import HeteroData

def build_hetero_data(nodes_csv: str, edges_csv: str, firm_feature_dim: int):
    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    data = HeteroData()

    # 1) 给每种节点类型建立 id -> local index
    type_to_nodes = {}
    id_maps = {}

    for ntype in nodes_df["node_type"].unique():
        sub = nodes_df[nodes_df["node_type"] == ntype].reset_index(drop=True)
        type_to_nodes[ntype] = sub
        id_maps[ntype] = {nid: i for i, nid in enumerate(sub["node_id"].tolist())}

        # 先用随机初始化特征；firm 节点后面可替换成 tab 特征
        x = torch.randn(len(sub), firm_feature_dim)
        data[ntype].x = x
        data[ntype].node_id = sub["node_id"].tolist()

    # 2) 建边
    for (src_type, rel, dst_type), sub in edges_df.groupby(["src_type", "relation", "dst_type"]):
        src_idx = [id_maps[src_type][sid] for sid in sub["src_id"].tolist()]
        dst_idx = [id_maps[dst_type][did] for did in sub["dst_id"].tolist()]
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        data[(src_type, rel, dst_type)].edge_index = edge_index

    return data