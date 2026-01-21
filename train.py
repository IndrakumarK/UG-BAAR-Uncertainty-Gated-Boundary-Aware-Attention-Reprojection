# train.py
import torch
from torch.utils.data import DataLoader
from models.ug_baar import UG_BAAR
from utils.losses import UGLoss
from utils.metrics import dice_score
from utils.seed import set_seed


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, uncertainty = model(x)
        loss = criterion(logits, y, uncertainty)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    scores = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1)
            scores.append(dice_score(probs, y))

    return sum(scores) / len(scores)


def main():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UG_BAAR().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = UGLoss()

    # NOTE: replace with real dataset loader
    train_loader = DataLoader([], batch_size=1)
    val_loader = DataLoader([], batch_size=1)

    train_epoch(model, train_loader, optimizer, criterion, device)
    validate(model, val_loader, device)


if __name__ == "__main__":
    main()
