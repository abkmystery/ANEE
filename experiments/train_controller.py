import os
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.append(SRC)

from anee.config import ANEEConfig
from anee.controller import LearnedController


def main():
    dataset_path = "datasets/traces.pt"
    save_path = "controllers/controller.pt"

    os.makedirs("controllers", exist_ok=True)

    data = torch.load(dataset_path)
    states = data["states"]
    actions = data["actions"]

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    config = ANEEConfig()
    controller = LearnedController(
        state_dim=config.state_dim,
        hidden_dim=config.controller_hidden_dim,
        total_layers=12,  # GPT-2 small
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(controller.parameters(), lr=1e-3)

    print("[ANEE] Training Controller...")

    for epoch in range(5):
        for batch_states, batch_actions in loader:
            optimizer.zero_grad()
            logits = controller.net(batch_states)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    torch.save(controller.state_dict(), save_path)
    print(f"[ANEE] Saved trained controller to {save_path}")


if __name__ == "__main__":
    main()
