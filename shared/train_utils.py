"""Shared training helpers for classifier notebooks."""

from __future__ import annotations

from typing import Iterable

import torch


def run_epoch(
    model: torch.nn.Module,
    loader: Iterable,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, list[int], list[int], list[float]]:
    train = optimizer is not None
    model.train(mode=train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []

    for x, pad_mask, state_idx, y in loader:
        x = x.to(device)
        pad_mask = pad_mask.to(device)
        state_idx = state_idx.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(x, state_idx, pad_mask)
            loss = criterion(logits, y)
            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        total_loss += float(loss.item()) * int(y.size(0))
        total_correct += int((preds == y).sum().item())
        total_count += int(y.size(0))
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())

    if total_count == 0:
        return 0.0, 0.0, y_true, y_pred, y_prob
    return total_loss / total_count, total_correct / total_count, y_true, y_pred, y_prob
