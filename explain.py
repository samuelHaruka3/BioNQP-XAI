import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import FirmDataset, collate_fn, TAB_FEATURES
from model import BioNQPXAI
from utils import classification_loss


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tab_x = batch["tab_x"].to(device)
            y_nqp = batch["y_nqp"].to(device)
            y_fin = batch["y_fin"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tab_x=tab_x,
                output_attentions=False,
            )
            loss = classification_loss(out, y_nqp, y_fin)
            total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FirmDataset(
        firms_csv="data/firms.csv",
        announcements_jsonl="data/announcements.jsonl",
        graph_edges_csv="data/graph_edges.csv",
        model_name="bert-base-chinese",
        max_length=256,
    )

    n_total = len(dataset)
    n_train = max(1, int(0.8 * n_total))
    n_val = max(1, n_total - n_train)
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = BioNQPXAI(
        model_name="bert-base-chinese",
        tab_in_dim=len(TAB_FEATURES),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_val = float("inf")
    save_path = Path("checkpoint.pt")

    for epoch in range(5):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        total_loss = 0.0

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tab_x = batch["tab_x"].to(device)
            y_nqp = batch["y_nqp"].to(device)
            y_fin = batch["y_fin"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                tab_x=tab_x,
                output_attentions=False,
            )
            loss = classification_loss(out, y_nqp, y_fin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = total_loss / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Training done.")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train()
