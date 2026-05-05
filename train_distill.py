#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mps_mod = getattr(torch.backends, "mps", None)
    if mps_mod is not None and mps_mod.is_available():
        torch.mps.manual_seed(seed)


def _mps_backend_flags() -> tuple[bool, bool]:
    mps_mod = getattr(torch.backends, "mps", None)
    if mps_mod is None:
        return False, False
    return bool(mps_mod.is_built()), bool(mps_mod.is_available())


def _pick_training_device() -> torch.device:
    mps_mod = getattr(torch.backends, "mps", None)
    mps_ok = mps_mod is not None and mps_mod.is_available()

    if sys.platform == "darwin" and mps_ok:
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


class DistillDataset(Dataset):
    def __init__(
        self,
        observations: np.ndarray,
        telemetry_speed_ids: np.ndarray,
        telemetry_longitudinal_ids: np.ndarray,
        telemetry_steering_ids: np.ndarray,
        telemetry_route_bin_ids: np.ndarray,
        telemetry_overspeed: np.ndarray,
        telemetry_off_lane: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.obs = torch.from_numpy(observations.astype(np.float32))
        self.speed = torch.from_numpy(telemetry_speed_ids.astype(np.int64))
        self.longitudinal = torch.from_numpy(telemetry_longitudinal_ids.astype(np.int64))
        self.steering = torch.from_numpy(telemetry_steering_ids.astype(np.int64))
        self.route_bin = torch.from_numpy(telemetry_route_bin_ids.astype(np.int64))
        self.overspeed = torch.from_numpy(telemetry_overspeed.astype(np.float32))
        self.off_lane = torch.from_numpy(telemetry_off_lane.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.obs[idx],
            self.speed[idx],
            self.longitudinal[idx],
            self.steering[idx],
            self.route_bin[idx],
            self.overspeed[idx],
            self.off_lane[idx],
            self.targets[idx],
        )


class DistillMLP(nn.Module):

    def __init__(
        self,
        *,
        obs_dim: int,
        speed_vocab_size: int,
        longitudinal_vocab_size: int,
        steering_vocab_size: int,
        route_bin_vocab_size: int,
        telemetry_embed_dim: int = 64,
        hidden_dim: int = 1024,
        num_hidden_blocks: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.speed_embed = nn.Embedding(num_embeddings=speed_vocab_size, embedding_dim=telemetry_embed_dim)
        self.longitudinal_embed = nn.Embedding(
            num_embeddings=longitudinal_vocab_size,
            embedding_dim=telemetry_embed_dim,
        )
        self.steering_embed = nn.Embedding(
            num_embeddings=steering_vocab_size,
            embedding_dim=telemetry_embed_dim,
        )
        self.route_bin_embed = nn.Embedding(
            num_embeddings=route_bin_vocab_size,
            embedding_dim=telemetry_embed_dim,
        )
        in_dim = obs_dim + (4 * telemetry_embed_dim) + 2
        blocks: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            blocks.append(nn.Dropout(dropout))
        for _ in range(num_hidden_blocks):
            blocks.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        blocks.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*blocks)

    def forward(
        self,
        obs: torch.Tensor,
        speed_ids: torch.Tensor,
        longitudinal_ids: torch.Tensor,
        steering_ids: torch.Tensor,
        route_bin_ids: torch.Tensor,
        overspeed: torch.Tensor,
        off_lane: torch.Tensor,
    ) -> torch.Tensor:
        s = self.speed_embed(speed_ids)
        l = self.longitudinal_embed(longitudinal_ids)
        st = self.steering_embed(steering_ids)
        r = self.route_bin_embed(route_bin_ids)
        flags = torch.stack([overspeed, off_lane], dim=-1)
        x = torch.cat([obs, s, l, st, r, flags], dim=-1)
        return self.net(x)


def _load_train_ready(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    with np.load(path, allow_pickle=True) as data:
        obs = np.asarray(data["observation_input_time_t"], dtype=np.float32)
        telemetry_fields = {
            "speed_id": np.asarray(data["telemetry_speed_id_t_plus_one"], dtype=np.int64),
            "longitudinal_id": np.asarray(
                data["telemetry_longitudinal_id_t_plus_one"],
                dtype=np.int64,
            ),
            "steering_id": np.asarray(data["telemetry_steering_id_t_plus_one"], dtype=np.int64),
            "route_bin_id": np.asarray(data["telemetry_route_bin_id_t_plus_one"], dtype=np.int64),
            "overspeed": np.asarray(data["telemetry_overspeed_t_plus_one"], dtype=np.float32),
            "off_lane": np.asarray(data["telemetry_off_lane_t_plus_one"], dtype=np.float32),
        }
        targets = np.asarray(data["expert_action_target_time_t"], dtype=np.float32)
        vocab_sizes = {
            "speed": int(np.asarray(data["telemetry_speed_vocab"], dtype=object).shape[0]),
            "longitudinal": int(
                np.asarray(data["telemetry_longitudinal_vocab"], dtype=object).shape[0]
            ),
            "steering": int(np.asarray(data["telemetry_steering_vocab"], dtype=object).shape[0]),
            "route_bin": int(np.asarray(data["telemetry_route_bin_vocab"], dtype=object).shape[0]),
        }
    return obs, telemetry_fields, targets, vocab_sizes


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)
    val_n = max(1, int(n * val_ratio))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    if train_idx.size == 0:
        train_idx = val_idx
    return train_idx, val_idx


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for obs, speed_id, longitudinal_id, steering_id, route_bin_id, overspeed, off_lane, targets in loader:
            obs = obs.to(device)
            speed_id = speed_id.to(device)
            longitudinal_id = longitudinal_id.to(device)
            steering_id = steering_id.to(device)
            route_bin_id = route_bin_id.to(device)
            overspeed = overspeed.to(device)
            off_lane = off_lane.to(device)
            targets = targets.to(device)
            pred = model(obs, speed_id, longitudinal_id, steering_id, route_bin_id, overspeed, off_lane)
            loss = criterion(pred, targets)
            batch_n = int(obs.shape[0])
            total_loss += float(loss.item()) * batch_n
            total += batch_n
    return total_loss / max(total, 1)


def main() -> None:
    data_path = Path(
        os.environ.get("METADRIVE_TRAIN_READY_NPZ", "data/distillation/train_ready.npz")
    )
    out_dir = Path(
        os.environ.get(
            "METADRIVE_DISTILL_CHECKPOINT_DIR",
            "data/distillation/checkpoints",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "distill_mlp.pt"
    out_model_last = out_dir / "distill_mlp_last.pt"
    out_metrics = out_dir / "distill_metrics.json"

    seed = _int_env("METADRIVE_DISTILL_SEED", 42)
    batch_size = _int_env("METADRIVE_DISTILL_BATCH_SIZE", 256)
    epochs = _int_env("METADRIVE_DISTILL_EPOCHS", 150)
    lr = _float_env("METADRIVE_DISTILL_LR", 1e-3)
    val_ratio = _float_env("METADRIVE_DISTILL_VAL_RATIO", 0.1)
    embed_dim = _int_env("METADRIVE_DISTILL_TELEMETRY_EMBED_DIM", 64)
    hidden_dim = _int_env("METADRIVE_DISTILL_HIDDEN_DIM", 1024)
    num_hidden_blocks = _int_env("METADRIVE_DISTILL_NUM_HIDDEN_BLOCKS", 6)
    dropout = _float_env("METADRIVE_DISTILL_DROPOUT", 0.15)
    weight_decay = _float_env("METADRIVE_DISTILL_WEIGHT_DECAY", 1e-4)
    grad_clip = _float_env("METADRIVE_DISTILL_GRAD_CLIP_NORM", 0.0)
    use_plateau = _bool_env("METADRIVE_DISTILL_USE_PLATEAU_SCHEDULER", True)
    plateau_patience = _int_env("METADRIVE_DISTILL_PLATEAU_PATIENCE", 6)
    plateau_factor = _float_env("METADRIVE_DISTILL_PLATEAU_FACTOR", 0.5)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing train-ready dataset: {data_path}. Run `python prepare_training_data.py` first."
        )

    _set_seed(seed)
    obs, telemetry_fields, targets, vocab_sizes = _load_train_ready(data_path)
    n, obs_dim = int(obs.shape[0]), int(obs.shape[1])
    if n < 2:
        raise ValueError(f"Need at least 2 training rows, got {n}.")
    if min(vocab_sizes.values()) < 1:
        raise ValueError(f"Telemetry vocab is empty in one or more fields: {vocab_sizes}")

    train_idx, val_idx = _split_indices(n, val_ratio=val_ratio, seed=seed)
    train_ds = DistillDataset(
        obs[train_idx],
        telemetry_fields["speed_id"][train_idx],
        telemetry_fields["longitudinal_id"][train_idx],
        telemetry_fields["steering_id"][train_idx],
        telemetry_fields["route_bin_id"][train_idx],
        telemetry_fields["overspeed"][train_idx],
        telemetry_fields["off_lane"][train_idx],
        targets[train_idx],
    )
    val_ds = DistillDataset(
        obs[val_idx],
        telemetry_fields["speed_id"][val_idx],
        telemetry_fields["longitudinal_id"][val_idx],
        telemetry_fields["steering_id"][val_idx],
        telemetry_fields["route_bin_id"][val_idx],
        telemetry_fields["overspeed"][val_idx],
        telemetry_fields["off_lane"][val_idx],
        targets[val_idx],
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    require_mps = _bool_env("METADRIVE_DISTILL_REQUIRE_MPS", True)
    mps_built, mps_available = _mps_backend_flags()
    device = _pick_training_device()
    print(
        f"Distillation training device: {device!s}  |  "
        f"cuda_available={torch.cuda.is_available()}  "
        f"mps_built={mps_built}  mps_available={mps_available}  |  "
        f"METADRIVE_DISTILL_REQUIRE_MPS={require_mps}",
        flush=True,
    )
    if require_mps and device.type != "mps":
        raise RuntimeError(
            "Refusing to start training: the selected device is "
            f"'{device.type}', not 'mps', while METADRIVE_DISTILL_REQUIRE_MPS is true (default). "
            "Use a PyTorch build with MPS on Apple Silicon, or set "
            "METADRIVE_DISTILL_REQUIRE_MPS=0 to allow CPU/CUDA training."
        )

    num_hidden_blocks = max(1, num_hidden_blocks)
    dropout = max(0.0, min(0.9, dropout))
    plateau_patience = max(1, plateau_patience)
    print(
        f"DistillMLP: embed_dim={embed_dim} hidden_dim={hidden_dim} "
        f"num_hidden_blocks={num_hidden_blocks} dropout={dropout}",
        flush=True,
    )
    print(
        f"Regularization: weight_decay={weight_decay}  "
        f"grad_clip_norm={grad_clip or 'off'}  "
        f"plateau_scheduler={use_plateau} (patience={plateau_patience})",
        flush=True,
    )

    model = DistillMLP(
        obs_dim=obs_dim,
        speed_vocab_size=vocab_sizes["speed"],
        longitudinal_vocab_size=vocab_sizes["longitudinal"],
        steering_vocab_size=vocab_sizes["steering"],
        route_bin_vocab_size=vocab_sizes["route_bin"],
        telemetry_embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_hidden_blocks=num_hidden_blocks,
        dropout=dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler: ReduceLROnPlateau | None = None
    if use_plateau:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=1e-6,
        )

    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for (
            batch_obs,
            batch_speed,
            batch_longitudinal,
            batch_steering,
            batch_route_bin,
            batch_overspeed,
            batch_off_lane,
            batch_targets,
        ) in train_loader:
            batch_obs = batch_obs.to(device)
            batch_speed = batch_speed.to(device)
            batch_longitudinal = batch_longitudinal.to(device)
            batch_steering = batch_steering.to(device)
            batch_route_bin = batch_route_bin.to(device)
            batch_overspeed = batch_overspeed.to(device)
            batch_off_lane = batch_off_lane.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(
                batch_obs,
                batch_speed,
                batch_longitudinal,
                batch_steering,
                batch_route_bin,
                batch_overspeed,
                batch_off_lane,
            )
            loss = criterion(pred, batch_targets)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_n = int(batch_obs.shape[0])
            running += float(loss.item()) * batch_n
            seen += batch_n

        train_loss = running / max(seen, 1)
        val_loss = _evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": float(train_loss),
                "val_mse": float(val_loss),
                "lr": lr_now,
            }
        )
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f} lr={lr_now:.2e}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "obs_dim": obs_dim,
                    "speed_vocab_size": vocab_sizes["speed"],
                    "longitudinal_vocab_size": vocab_sizes["longitudinal"],
                    "steering_vocab_size": vocab_sizes["steering"],
                    "route_bin_vocab_size": vocab_sizes["route_bin"],
                    "telemetry_embed_dim": embed_dim,
                    "hidden_dim": hidden_dim,
                    "num_hidden_blocks": num_hidden_blocks,
                    "dropout": dropout,
                },
                out_model,
            )

    _ckpt = {
        "model_state_dict": model.state_dict(),
        "obs_dim": obs_dim,
        "speed_vocab_size": vocab_sizes["speed"],
        "longitudinal_vocab_size": vocab_sizes["longitudinal"],
        "steering_vocab_size": vocab_sizes["steering"],
        "route_bin_vocab_size": vocab_sizes["route_bin"],
        "telemetry_embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_hidden_blocks": num_hidden_blocks,
        "dropout": dropout,
    }
    torch.save(_ckpt, out_model_last)

    metrics = {
        "data_path": str(data_path),
        "num_rows": n,
        "obs_dim": obs_dim,
        "speed_vocab_size": vocab_sizes["speed"],
        "longitudinal_vocab_size": vocab_sizes["longitudinal"],
        "steering_vocab_size": vocab_sizes["steering"],
        "route_bin_vocab_size": vocab_sizes["route_bin"],
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "telemetry_embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_hidden_blocks": num_hidden_blocks,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip,
        "use_plateau_scheduler": use_plateau,
        "best_val_mse": float(best_val),
        "history": history,
        "checkpoint_path_best_val": str(out_model),
        "checkpoint_path_last_epoch": str(out_model_last),
        "checkpoint_path": str(out_model),
        "device": str(device),
    }
    out_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Saved best-val checkpoint: {out_model}", flush=True)
    print(f"Saved last-epoch checkpoint: {out_model_last}", flush=True)
    print(f"Saved metrics: {out_metrics}", flush=True)


if __name__ == "__main__":
    main()
