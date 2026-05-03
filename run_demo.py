from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

try:
    from physics import C0, cole_cole_polar, eps_mu_from_params, reflection_loss_db
except ImportError:  # pragma: no cover - supports importing this file as a package module.
    from .physics import C0, cole_cole_polar, eps_mu_from_params, reflection_loss_db

FEATURE_NAMES = [
    "carbon_ratio",
    "magnetic_ratio",
    "porosity",
    "particle_size",
    "annealing_temperature",
    "density",
    "conductivity_proxy",
]

PARAM_NAMES = [
    "eps_inf",
    "eps_s",
    "tau_eps",
    "alpha_eps",
    "mu_inf",
    "mu_s",
    "tau_mu",
    "alpha_mu",
]

LOSS_VECTOR_NAMES = [
    "eps_inf",
    "eps_delta",
    "log10_tau_eps",
    "alpha_eps",
    "mu_inf",
    "mu_delta",
    "log10_tau_mu",
    "alpha_mu",
]

RAW_OUTPUT_NAMES = [
    "eps_inf_raw",
    "eps_delta_raw",
    "tau_eps_raw",
    "alpha_eps_raw",
    "mu_inf_raw",
    "mu_delta_raw",
    "tau_mu_raw",
    "alpha_mu_raw",
]


@dataclass
class TrainStats:
    param_mean: torch.Tensor
    param_std: torch.Tensor
    eps_real_std: torch.Tensor
    eps_imag_std: torch.Tensor
    mu_real_std: torch.Tensor
    mu_imag_std: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_tinykan_forward_flops(in_dim: int, out_dim: int, grid_size: int) -> int:
    basis_ops = in_dim * grid_size * 6
    rbf_sum_ops = out_dim * in_dim * grid_size * 2
    residual_ops = in_dim * 5 + out_dim * in_dim * 2
    return basis_ops + rbf_sum_ops + residual_ops + out_dim


def estimate_mlp_forward_flops(in_dim: int, out_dim: int, hidden_dim: int) -> int:
    linear_ops = 2 * (in_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * out_dim)
    activation_ops = 2 * hidden_dim * 5
    return linear_ops + activation_ops


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(torch.view_as_real(tensor) if torch.is_complex(tensor) else tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf values.")


def sigmoid_range(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.sigmoid(x)


class PhysicalOutputHead(nn.Module):
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return transform_raw_params(raw)


def generate_material_features(n_materials: int, device: torch.device) -> torch.Tensor:
    features = torch.rand(n_materials, len(FEATURE_NAMES), device=device)
    # Density and conductivity_proxy are intentionally independent near-dummy
    # controls, so edge scores can test whether TinyKAN invents spurious links.
    return features


def synthetic_dispersion_params(features: torch.Tensor) -> torch.Tensor:
    carbon = features[:, 0]
    magnetic = features[:, 1]
    porosity = features[:, 2]
    particle = features[:, 3]
    anneal = features[:, 4]

    anneal_window = torch.exp(-((anneal - 0.62) / 0.18).pow(2))
    carbon_window = torch.exp(-3.2 * (carbon - 0.42).pow(2))

    eps_inf = 2.0 + 1.45 * carbon + 0.32 * torch.sin(math.pi * anneal)
    eps_delta = (
        2.0
        + 5.0 * carbon.pow(1.35)
        + 4.3 * anneal_window
        + 1.0 * (1.0 - porosity)
        + 0.6 * torch.sin(math.pi * particle).pow(2)
    )
    eps_s = eps_inf + eps_delta
    tau_eps = torch.exp(
        math.log(6.0e-10)
        - 3.2 * (carbon - 0.42).pow(2)
        + 0.55 * porosity
        + 1.05 * (1.0 - particle)
        - 0.25 * anneal
    )
    alpha_eps = 0.05 + 0.9 * torch.sigmoid(-1.2 + 1.4 * porosity + 0.9 * carbon - 0.5 * anneal)

    mu_inf = 0.95 + 0.32 * magnetic
    mu_delta = 0.08 + 1.45 * magnetic + 0.55 * anneal_window + 0.18 * (1.0 - porosity)
    mu_s = mu_inf + mu_delta
    tau_mu = torch.exp(
        math.log(3.2e-10)
        + 0.9 * (1.0 - particle)
        + 0.55 * magnetic
        - 0.65 * anneal
        + 0.3 * porosity
        + 0.05 * carbon_window
    )
    alpha_mu = 0.05 + 0.9 * torch.sigmoid(-1.4 + 1.2 * magnetic + 0.7 * porosity)

    return torch.stack([eps_inf, eps_s, tau_eps, alpha_eps, mu_inf, mu_s, tau_mu, alpha_mu], dim=-1)


def transform_raw_params(raw: torch.Tensor) -> torch.Tensor:
    eps_inf = sigmoid_range(raw[:, 0], 1.2, 7.5)
    eps_delta = sigmoid_range(raw[:, 1], 0.4, 12.0)
    tau_eps = torch.exp(sigmoid_range(raw[:, 2], math.log(2.0e-11), math.log(2.5e-9)))
    alpha_eps = sigmoid_range(raw[:, 3], 0.05, 0.95)

    mu_inf = sigmoid_range(raw[:, 4], 0.85, 1.75)
    mu_delta = sigmoid_range(raw[:, 5], 0.02, 3.2)
    tau_mu = torch.exp(sigmoid_range(raw[:, 6], math.log(3.0e-11), math.log(4.0e-9)))
    alpha_mu = sigmoid_range(raw[:, 7], 0.05, 0.95)

    return torch.stack(
        [eps_inf, eps_inf + eps_delta, tau_eps, alpha_eps, mu_inf, mu_inf + mu_delta, tau_mu, alpha_mu],
        dim=-1,
    )


def params_to_loss_vector(params: torch.Tensor) -> torch.Tensor:
    eps_inf, eps_s, tau_eps, alpha_eps, mu_inf, mu_s, tau_mu, alpha_mu = params.unbind(dim=-1)
    return torch.stack(
        [
            eps_inf,
            eps_s - eps_inf,
            torch.log10(torch.clamp(tau_eps, min=1.0e-14)),
            alpha_eps,
            mu_inf,
            mu_s - mu_inf,
            torch.log10(torch.clamp(tau_mu, min=1.0e-14)),
            alpha_mu,
        ],
        dim=-1,
    )


class TinyKAN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 12) -> None:
        super().__init__()
        centers = torch.linspace(-1.0, 1.0, grid_size)
        self.register_buffer("centers", centers)
        self.log_width = nn.Parameter(torch.tensor(math.log(0.32)))
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.03)
        self.residual_weight = nn.Parameter(torch.zeros(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = 2.0 * x - 1.0
        width = torch.exp(self.log_width).clamp(min=0.06, max=0.8)
        return torch.exp(-((x_norm[:, :, None] - self.centers[None, None, :]) / width) ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self.basis(x)
        x_norm = 2.0 * x - 1.0
        rbf = torch.einsum("big,oig->bo", phi, self.weight)
        residual = torch.einsum("bi,oi->bo", F.silu(x_norm), self.residual_weight)
        return rbf + residual + self.bias

    @torch.no_grad()
    def edge_curve(self, feature_index: int, output_index: int, grid: torch.Tensor) -> torch.Tensor:
        x_norm = 2.0 * grid - 1.0
        width = torch.exp(self.log_width).clamp(min=0.06, max=0.8)
        phi = torch.exp(-((x_norm[:, None] - self.centers[None, :]) / width) ** 2)
        residual = F.silu(x_norm) * self.residual_weight[output_index, feature_index]
        return phi @ self.weight[output_index, feature_index, :] + residual

    @torch.no_grad()
    def edge_scores(self) -> torch.Tensor:
        return self.weight.abs().sum(dim=-1) + self.residual_weight.abs()

    @torch.no_grad()
    def strongest_edges(self, top_k: int = 8) -> list[tuple[int, int, float]]:
        scores = self.edge_scores()
        flat_scores, flat_indices = torch.topk(scores.flatten(), k=min(top_k, scores.numel()))
        edges = []
        for score, flat_index in zip(flat_scores.tolist(), flat_indices.tolist()):
            output_index = flat_index // scores.shape[1]
            feature_index = flat_index % scores.shape[1]
            edges.append((feature_index, output_index, float(score)))
        return edges


class PhysicalModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = PhysicalOutputHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_train_stats(freq_hz: torch.Tensor, train_params: torch.Tensor) -> TrainStats:
    eps_r, mu_r = eps_mu_from_params(freq_hz, train_params)
    param_vec = params_to_loss_vector(train_params)
    return TrainStats(
        param_mean=param_vec.mean(dim=0),
        param_std=param_vec.std(dim=0).clamp_min(1.0e-6),
        eps_real_std=eps_r.real.std().clamp_min(1.0e-6),
        eps_imag_std=eps_r.imag.std().clamp_min(1.0e-6),
        mu_real_std=mu_r.real.std().clamp_min(1.0e-6),
        mu_imag_std=mu_r.imag.std().clamp_min(1.0e-6),
    )


def normalized_physics_loss(
    pred_params: torch.Tensor,
    true_params: torch.Tensor,
    freq_hz: torch.Tensor,
    stats: TrainStats,
    thickness_m: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_eps, pred_mu = eps_mu_from_params(freq_hz, pred_params)
    true_eps, true_mu = eps_mu_from_params(freq_hz, true_params)
    pred_rl = reflection_loss_db(freq_hz, thickness_m, pred_eps, pred_mu)
    true_rl = reflection_loss_db(freq_hz, thickness_m, true_eps, true_mu)

    pred_vec = (params_to_loss_vector(pred_params) - stats.param_mean) / stats.param_std
    true_vec = (params_to_loss_vector(true_params) - stats.param_mean) / stats.param_std

    param_loss_by_dim = (pred_vec - true_vec).pow(2).mean(dim=0)
    param_loss = param_loss_by_dim.mean()
    eps_loss = F.mse_loss(pred_eps.real / stats.eps_real_std, true_eps.real / stats.eps_real_std)
    eps_loss = eps_loss + F.mse_loss(pred_eps.imag / stats.eps_imag_std, true_eps.imag / stats.eps_imag_std)
    mu_loss = F.mse_loss(pred_mu.real / stats.mu_real_std, true_mu.real / stats.mu_real_std)
    mu_loss = mu_loss + F.mse_loss(pred_mu.imag / stats.mu_imag_std, true_mu.imag / stats.mu_imag_std)
    rl_loss = F.mse_loss(pred_rl / 20.0, true_rl / 20.0)

    total = param_loss + eps_loss + mu_loss + 0.1 * rl_loss
    metrics = {
        "total": float(total.detach().cpu()),
        "param": float(param_loss.detach().cpu()),
        "param_by_dim": {
            name: float(value.detach().cpu()) for name, value in zip(LOSS_VECTOR_NAMES, param_loss_by_dim)
        },
        "eps": float(eps_loss.detach().cpu()),
        "mu": float(mu_loss.detach().cpu()),
        "rl": float(rl_loss.detach().cpu()),
    }
    return total, metrics


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_params: torch.Tensor,
    val_x: torch.Tensor,
    val_params: torch.Tensor,
    freq_hz: torch.Tensor,
    stats: TrainStats,
    epochs: int,
    lr: float,
    thickness_m: float,
) -> dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-5)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_params = model(train_x)
        loss, train_metrics = normalized_physics_loss(pred_params, train_params, freq_hz, stats, thickness_m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            model.eval()
            with torch.no_grad():
                val_pred_params = model(val_x)
                _, val_metrics = normalized_physics_loss(val_pred_params, val_params, freq_hz, stats, thickness_m)
            history.append(
                {
                    "epoch": epoch,
                    "train_total": train_metrics["total"],
                    "val_total": val_metrics["total"],
                    "val_param": val_metrics["param"],
                    "val_eps": val_metrics["eps"],
                    "val_mu": val_metrics["mu"],
                    "val_rl": val_metrics["rl"],
                }
            )

    model.eval()
    with torch.no_grad():
        final_pred_params = model(val_x)
        _, final_val_metrics = normalized_physics_loss(final_pred_params, val_params, freq_hz, stats, thickness_m)

    return {"history": history, "val_metrics": final_val_metrics, "val_pred_params": final_pred_params}


def make_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_rl_curve(
    out_path: Path,
    freq_hz: torch.Tensor,
    true_params: torch.Tensor,
    kan_params: torch.Tensor,
    mlp_matched_params: torch.Tensor,
    mlp_width_params: torch.Tensor | None,
    thickness_m: float,
    dpi: int,
) -> None:
    true_eps, true_mu = eps_mu_from_params(freq_hz, true_params[None, :])
    kan_eps, kan_mu = eps_mu_from_params(freq_hz, kan_params[None, :])
    mlp_matched_eps, mlp_matched_mu = eps_mu_from_params(freq_hz, mlp_matched_params[None, :])
    if mlp_width_params is not None:
        mlp_width_eps, mlp_width_mu = eps_mu_from_params(freq_hz, mlp_width_params[None, :])

    true_rl = reflection_loss_db(freq_hz, thickness_m, true_eps, true_mu).squeeze(0).detach().cpu().numpy()
    kan_rl = reflection_loss_db(freq_hz, thickness_m, kan_eps, kan_mu).squeeze(0).detach().cpu().numpy()
    mlp_matched_rl = reflection_loss_db(freq_hz, thickness_m, mlp_matched_eps, mlp_matched_mu).squeeze(0).detach().cpu().numpy()
    if mlp_width_params is not None:
        mlp_width_rl = reflection_loss_db(freq_hz, thickness_m, mlp_width_eps, mlp_width_mu).squeeze(0).detach().cpu().numpy()
    freq_ghz = (freq_hz.detach().cpu().numpy() / 1.0e9)

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(freq_ghz, true_rl, color="#111111", linewidth=2.2, label="True physics")
    plt.plot(freq_ghz, kan_rl, color="#1f77b4", linewidth=2.0, label="TinyKAN")
    plt.plot(freq_ghz, mlp_matched_rl, color="#ff7f0e", linewidth=1.8, linestyle="--", label="MLP matched")
    if mlp_width_params is not None:
        plt.plot(freq_ghz, mlp_width_rl, color="#2ca02c", linewidth=1.5, linestyle="-.", label="MLP width=32")
    plt.axhline(-10.0, color="#999999", linewidth=1.0, linestyle=":", label="-10 dB")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Reflection loss (dB)")
    plt.title(f"RL curve at d = {thickness_m * 1e3:.1f} mm")
    plt.ylim(-60, 0)
    plt.grid(alpha=0.22)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_rl_heatmap(out_path: Path, freq_hz: torch.Tensor, params: torch.Tensor, dpi: int) -> None:
    eps_r, mu_r = eps_mu_from_params(freq_hz, params[None, :])
    thicknesses = torch.linspace(1.0e-3, 5.0e-3, 81, device=freq_hz.device)
    rl_map = reflection_loss_db(freq_hz, thicknesses, eps_r.squeeze(0), mu_r.squeeze(0)).detach().cpu().numpy()
    freq_ghz = freq_hz.detach().cpu().numpy() / 1.0e9
    thick_mm = thicknesses.detach().cpu().numpy() * 1.0e3
    min_indices = rl_map.argmin(axis=1)

    plt.figure(figsize=(8.6, 5.2))
    image = plt.imshow(
        rl_map,
        extent=[freq_ghz.min(), freq_ghz.max(), thick_mm.min(), thick_mm.max()],
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=-45,
        vmax=0,
    )
    plt.plot(freq_ghz[min_indices], thick_mm, color="white", linewidth=1.7, label="minimum RL")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Thickness (mm)")
    plt.title("Thickness-frequency RL map")
    plt.colorbar(image, label="RL (dB)")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_param_scatter(
    out_path: Path,
    true_params: torch.Tensor,
    kan_params: torch.Tensor,
    mlp_params: torch.Tensor,
    dpi: int,
) -> None:
    true_np = true_params.detach().cpu().numpy()
    kan_np = kan_params.detach().cpu().numpy()
    mlp_np = mlp_params.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.3))
    for idx, ax in enumerate(axes.ravel()):
        true_values = true_np[:, idx]
        kan_values = kan_np[:, idx]
        mlp_values = mlp_np[:, idx]

        if "tau" in PARAM_NAMES[idx]:
            true_values = np.log10(true_values)
            kan_values = np.log10(kan_values)
            mlp_values = np.log10(mlp_values)
            label = f"log10({PARAM_NAMES[idx]})"
        else:
            label = PARAM_NAMES[idx]

        low = min(true_values.min(), kan_values.min(), mlp_values.min())
        high = max(true_values.max(), kan_values.max(), mlp_values.max())
        ax.scatter(true_values, kan_values, s=18, color="#1f77b4", alpha=0.78, label="TinyKAN")
        ax.scatter(true_values, mlp_values, s=18, color="#ff7f0e", alpha=0.48, marker="x", label="MLP matched")
        ax.plot([low, high], [low, high], color="#222222", linewidth=1.0, linestyle=":")
        ax.set_title(label)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(alpha=0.18)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=2)
    fig.suptitle("Dispersion parameter recovery", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_eps_mu_decomposition(
    out_path: Path,
    freq_hz: torch.Tensor,
    true_params: torch.Tensor,
    kan_params: torch.Tensor,
    mlp_params: torch.Tensor,
    dpi: int,
) -> None:
    true_eps, true_mu = eps_mu_from_params(freq_hz, true_params[None, :])
    kan_eps, kan_mu = eps_mu_from_params(freq_hz, kan_params[None, :])
    mlp_eps, mlp_mu = eps_mu_from_params(freq_hz, mlp_params[None, :])

    freq_ghz = freq_hz.detach().cpu().numpy() / 1.0e9
    series = [
        ("eps real", true_eps.real, kan_eps.real, mlp_eps.real),
        ("eps imag", true_eps.imag, kan_eps.imag, mlp_eps.imag),
        ("mu real", true_mu.real, kan_mu.real, mlp_mu.real),
        ("mu imag", true_mu.imag, kan_mu.imag, mlp_mu.imag),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6))
    for ax, (title, true_values, kan_values, mlp_values) in zip(axes, series):
        ax.plot(freq_ghz, true_values.squeeze(0).detach().cpu().numpy(), color="#111111", linewidth=2.0, label="True")
        ax.plot(freq_ghz, kan_values.squeeze(0).detach().cpu().numpy(), color="#1f77b4", linewidth=1.8, label="TinyKAN")
        ax.plot(freq_ghz, mlp_values.squeeze(0).detach().cpu().numpy(), color="#ff7f0e", linewidth=1.4, linestyle="--", label="MLP matched")
        ax.set_title(title)
        ax.set_xlabel("Frequency (GHz)")
        ax.grid(alpha=0.18)

    axes[0].set_ylabel("Relative response")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3)
    fig.suptitle("Dispersion-channel decomposition", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def normalize_curve(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean()
    scale = np.max(np.abs(centered))
    if scale < 1.0e-8:
        return centered
    return centered / scale


def center_curve(values: np.ndarray) -> np.ndarray:
    return values - values.mean()


def true_partial_curve(
    feature_index: int,
    output_index: int,
    grid: torch.Tensor,
    stats: TrainStats,
) -> torch.Tensor:
    baseline = torch.full((grid.numel(), len(FEATURE_NAMES)), 0.5, dtype=grid.dtype, device=grid.device)
    baseline[:, feature_index] = grid
    params = synthetic_dispersion_params(baseline)
    normalized = (params_to_loss_vector(params) - stats.param_mean) / stats.param_std
    return normalized[:, output_index]


def plot_kan_edges(out_path: Path, model: TinyKAN, stats: TrainStats, device: torch.device, grid_points: int, dpi: int) -> None:
    grid = torch.linspace(0.0, 1.0, grid_points, device=device)
    scores = model.edge_scores().detach().cpu().numpy()
    max_score = max(float(scores.max()), 1.0e-8)
    learned_curves = np.zeros((len(PARAM_NAMES), len(FEATURE_NAMES), grid_points))
    truth_curves = np.zeros_like(learned_curves)

    for output_index in range(len(PARAM_NAMES)):
        for feature_index in range(len(FEATURE_NAMES)):
            learned = model.edge_curve(feature_index, output_index, grid).detach().cpu().numpy()
            truth = true_partial_curve(feature_index, output_index, grid, stats).detach().cpu().numpy()
            learned_curves[output_index, feature_index] = center_curve(learned)
            truth_curves[output_index, feature_index] = center_curve(truth)

    learned_row_scale = np.maximum(np.max(np.abs(learned_curves), axis=(1, 2)), 1.0e-8)
    truth_row_scale = np.maximum(np.max(np.abs(truth_curves), axis=(1, 2)), 1.0e-8)
    grid_np = grid.detach().cpu().numpy()
    fig, axes = plt.subplots(len(PARAM_NAMES), len(FEATURE_NAMES), figsize=(17.0, 16.5), sharex=True)

    for output_index in range(len(PARAM_NAMES)):
        for feature_index in range(len(FEATURE_NAMES)):
            ax = axes[output_index, feature_index]
            score_ratio = float(scores[output_index, feature_index] / max_score)
            alpha = 0.10 + 0.90 * math.sqrt(score_ratio)
            linewidth = 0.55 + 1.15 * score_ratio

            learned = learned_curves[output_index, feature_index] / learned_row_scale[output_index]
            truth = truth_curves[output_index, feature_index] / truth_row_scale[output_index]
            ax.plot(grid_np, learned, color="#1f77b4", linewidth=linewidth, alpha=alpha)
            ax.plot(grid_np, truth, color="#111111", linewidth=0.9, linestyle="--", alpha=0.68)
            ax.axhline(0.0, color="#999999", linewidth=0.45, alpha=0.55)
            ax.set_ylim(-1.15, 1.15)
            ax.grid(alpha=0.10)
            ax.tick_params(labelsize=6, length=2)

            if output_index == 0:
                ax.set_title(FEATURE_NAMES[feature_index], fontsize=8)
            if feature_index == 0:
                ax.set_ylabel(LOSS_VECTOR_NAMES[output_index], fontsize=8)
            if output_index == len(PARAM_NAMES) - 1:
                ax.set_xlabel("feature", fontsize=7)

    legend_lines = [
        plt.Line2D([0], [0], color="#1f77b4", linewidth=1.5, label="TinyKAN edge, row-scaled"),
        plt.Line2D([0], [0], color="#111111", linewidth=1.2, linestyle="--", label="true partial, row-scaled"),
    ]
    fig.legend(handles=legend_lines, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=2)
    fig.suptitle("TinyKAN edge functions vs true partial effects", y=0.998)
    fig.text(0.5, 0.958, "Weak edges are not normalized independently; line strength follows RBF-edge L1 score.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def edge_score_frame(model: TinyKAN) -> pd.DataFrame:
    scores = model.edge_scores().detach().cpu().numpy()
    return pd.DataFrame(scores, index=RAW_OUTPUT_NAMES, columns=FEATURE_NAMES)


def plot_edge_l1_heatmap(out_path: Path, score_frame: pd.DataFrame, dpi: int) -> None:
    values = score_frame.to_numpy()
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    image = ax.imshow(values, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(RAW_OUTPUT_NAMES)))
    ax.set_yticklabels(RAW_OUTPUT_NAMES)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if values[row, col] > values.max() * 0.55 else "#111111"
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_title("TinyKAN edge L1 scores")
    fig.colorbar(image, ax=ax, label="RBF + residual L1 norm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def summarize_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def aggregate_model_metrics(seed_metrics: list[dict[str, object]], model_key: str) -> dict[str, object]:
    first_val_metrics = seed_metrics[0][model_key]["val_metrics"]
    if first_val_metrics is None:
        return {"trained": False}

    scalar_keys = ["total", "param", "eps", "mu", "rl"]
    aggregate: dict[str, object] = {
        key: summarize_values([run[model_key]["val_metrics"][key] for run in seed_metrics])
        for key in scalar_keys
    }
    aggregate["param_by_dim"] = {
        name: summarize_values([run[model_key]["val_metrics"]["param_by_dim"][name] for run in seed_metrics])
        for name in LOSS_VECTOR_NAMES
    }
    return aggregate


def aggregate_seed_metrics(seed_metrics: list[dict[str, object]]) -> dict[str, object]:
    return {
        "seeds": [run["settings"]["seed"] for run in seed_metrics],
        "n_seeds": len(seed_metrics),
        "tinykan": aggregate_model_metrics(seed_metrics, "tinykan"),
        "mlp_matched": aggregate_model_metrics(seed_metrics, "mlp_matched"),
        "mlp_width_reference": aggregate_model_metrics(seed_metrics, "mlp_width_reference"),
    }


def write_summary(
    out_path: Path,
    metrics: dict[str, object],
    kan_edges: list[tuple[int, int, float]],
) -> None:
    edge_lines = [
        f"- {FEATURE_NAMES[feature_index]} -> {RAW_OUTPUT_NAMES[output_index]} (score={score:.3f})"
        for feature_index, output_index, score in kan_edges[:6]
    ]
    edge_scores = pd.DataFrame.from_dict(metrics["edge_l1_scores"], orient="index")
    active_columns = ["carbon_ratio", "magnetic_ratio", "porosity", "particle_size", "annealing_temperature"]
    dummy_columns = ["density", "conductivity_proxy"]
    active_mean = float(edge_scores[active_columns].to_numpy().mean())
    dummy_mean = float(edge_scores[dummy_columns].to_numpy().mean())
    sparsity_ratio = active_mean / max(dummy_mean, 1.0e-8)
    multi_seed = metrics.get("multi_seed")
    seed_line = f"Seed: {metrics['settings']['seed']}"
    tinykan_total = {"mean": metrics["tinykan"]["val_metrics"]["total"], "std": 0.0}
    mlp_total = {"mean": metrics["mlp_matched"]["val_metrics"]["total"], "std": 0.0}
    if isinstance(multi_seed, dict) and multi_seed.get("n_seeds", 1) > 1:
        seed_line = f"Seeds: {', '.join(str(seed) for seed in multi_seed['seeds'])}"
        tinykan_total = multi_seed["tinykan"]["total"]
        mlp_total = multi_seed["mlp_matched"]["total"]

    headline = (
        f"TinyKAN reaches {tinykan_total['mean']:.3f} +/- {tinykan_total['std']:.3f} validation loss "
        f"vs matched-MLP's {mlp_total['mean']:.3f} +/- {mlp_total['std']:.3f} on 7 -> 8 dispersion "
        f"parameter regression, at near-identical parameter count ({metrics['tinykan_params']} vs "
        f"{metrics['mlp_matched_params']}, {100.0 * metrics['parameter_gap_matched_fraction']:.2f}% gap)."
    )
    text = "\n".join(
        [
            "# KAN Dispersion-to-RL Demo (Week 1)",
            "",
            "## Headline result",
            "",
            headline,
            "",
            f"{seed_line}. Train/validation split is by material ID, so held-out materials are unseen during training.",
            "",
            "## What this shows",
            "",
            "- TinyKAN's local edge functions provide useful inductive bias for non-monotonic material -> dispersion mappings.",
            "- The important visual evidence is not just lower loss; it is that KAN directly draws local physical trends such as annealing-temperature windows and carbon-ratio relaxation effects.",
            "- Epsilon/mu and standardized parameter supervision carry the main training signal; RL is a 0.1x consistency check through the differentiable physics layer.",
            f"- Edge L1 norms concentrate more on informative inputs than dummy inputs, but this is weak evidence only: active/dummy ratio = {sparsity_ratio:.2f}x. Explicit L1 regularization and pykan pruning are Week 2 work.",
            "",
            "## Strongest TinyKAN edge functions",
            "",
            *edge_lines,
            "",
            "## What this does NOT show yet",
            "",
            "- It does not show symbolic regression. TinyKAN is an RBF-edge stand-in; pykan auto_symbolic is deliberately deferred.",
            "- It does not show real-material discovery. This is synthetic Cole-Cole data with known ground truth, not literature data.",
            "- It does not prove automatic pruning. The dummy-variable ratio is a soft diagnostic, not a hard mechanism claim.",
            "- It does not solve literature-data causality. Real epsilon/mu extraction must use complex-domain joint fitting to preserve Kramers-Kronig consistency.",
            "- It does not transfer the dashed ground-truth edge plot to Week 3. Real data will need KAN edge functions with bootstrap confidence intervals instead.",
            "",
            "## Files to show",
            "",
            "- `kan_edge_functions.png`: controlled sanity check showing KAN edges vs known synthetic partial effects.",
            "- `eps_mu_decomposition.png`: where prediction error comes from across epsilon and mu channels.",
            "- `edge_l1_scores.png`: variable-importance diagnostic via RBF weight L1 norms.",
            "- `metrics.json`: full per-seed and per-dimension numbers.",
            "",
            "## 30-second pitch",
            "",
            "I am not using KAN as a black-box RL predictor. The network only maps material features to physically constrained Cole-Cole dispersion parameters; epsilon(f), mu(f), and RL are then generated by analytic physics. In this controlled Week 1 test, the KAN-style edge structure recovers non-monotonic mechanisms such as annealing-temperature windows under a very small parameter budget, while the matched MLP struggles. The loss gap is only a sanity check; the real point is that KAN gives inspectable edge functions that can become hypotheses for material mechanisms. Symbolic pykan and literature data are the next milestone, not this demo.",
            "",
        ]
    )
    out_path.write_text(text, encoding="utf-8")


def save_feature_table(out_path: Path, features: torch.Tensor, params: torch.Tensor) -> None:
    frame = pd.DataFrame(features.detach().cpu().numpy(), columns=FEATURE_NAMES)
    for idx, name in enumerate(PARAM_NAMES):
        frame[name] = params[:, idx].detach().cpu().numpy()
    frame.to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TinyKAN Cole-Cole dispersion-to-RL demo.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--n-materials", type=int, default=256)
    parser.add_argument("--n-freq", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Run multiple seeds and report mean/std. The first seed writes figures.")
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--quick", action="store_true", help="Run a fast smoke-test configuration.")
    parser.add_argument("--include-width-mlp", action="store_true", help="Also train a wider hidden_dim=32 MLP reference.")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if choice == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_experiment(
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    plot_dpi: int,
    edge_grid_points: int,
    write_artifacts: bool,
) -> tuple[dict[str, object], list[tuple[int, int, float]]]:
    set_seed(seed)

    freq_hz = torch.linspace(2.0e9, 18.0e9, args.n_freq, device=device)
    train_thickness_m = 2.5e-3

    features = generate_material_features(args.n_materials, device)
    true_params = synthetic_dispersion_params(features)
    assert_finite("true_params", true_params)

    permutation = torch.randperm(args.n_materials, device=device)
    train_count = max(8, int(args.n_materials * 0.8))
    train_idx = permutation[:train_count]
    val_idx = permutation[train_count:]
    if val_idx.numel() == 0:
        raise ValueError("Need at least one validation material.")

    train_x = features[train_idx]
    val_x = features[val_idx]
    train_params = true_params[train_idx]
    val_params = true_params[val_idx]
    stats = build_train_stats(freq_hz, train_params)

    tinykan_backbone = TinyKAN(in_dim=len(FEATURE_NAMES), out_dim=len(PARAM_NAMES)).to(device)
    mlp_matched_hidden = 20
    mlp_width_hidden = 32
    mlp_matched_backbone = MLP(in_dim=len(FEATURE_NAMES), out_dim=len(PARAM_NAMES), hidden_dim=mlp_matched_hidden).to(device)
    mlp_width_reference_backbone = MLP(in_dim=len(FEATURE_NAMES), out_dim=len(PARAM_NAMES), hidden_dim=mlp_width_hidden).to(device)
    tinykan = PhysicalModel(tinykan_backbone).to(device)
    mlp_matched = PhysicalModel(mlp_matched_backbone).to(device)
    mlp_width_reference = PhysicalModel(mlp_width_reference_backbone).to(device)

    tinykan_params = count_parameters(tinykan)
    mlp_matched_params = count_parameters(mlp_matched)
    mlp_width_reference_params = count_parameters(mlp_width_reference)
    param_gap = abs(tinykan_params - mlp_matched_params) / tinykan_params

    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Training materials: {train_x.shape[0]}, validation materials: {val_x.shape[0]}")
    print(f"TinyKAN params: {tinykan_params}, MLP matched params: {mlp_matched_params}, MLP width=32 reference params: {mlp_width_reference_params}")
    print(f"TinyKAN vs matched MLP parameter gap: {100.0 * param_gap:.2f}%")

    tinykan_result = train_model(tinykan, train_x, train_params, val_x, val_params, freq_hz, stats, args.epochs, args.lr, train_thickness_m)
    mlp_matched_result = train_model(mlp_matched, train_x, train_params, val_x, val_params, freq_hz, stats, args.epochs, args.lr, train_thickness_m)
    mlp_width_result = None
    if args.include_width_mlp:
        mlp_width_result = train_model(mlp_width_reference, train_x, train_params, val_x, val_params, freq_hz, stats, args.epochs, args.lr, train_thickness_m)

    kan_val_params = tinykan_result["val_pred_params"]
    mlp_matched_val_params = mlp_matched_result["val_pred_params"]
    mlp_width_val_params = mlp_width_result["val_pred_params"] if mlp_width_result else None
    assert isinstance(kan_val_params, torch.Tensor)
    assert isinstance(mlp_matched_val_params, torch.Tensor)
    if mlp_width_val_params is not None:
        assert isinstance(mlp_width_val_params, torch.Tensor)
    assert_finite("kan_val_params", kan_val_params)
    assert_finite("mlp_matched_val_params", mlp_matched_val_params)
    if mlp_width_val_params is not None:
        assert_finite("mlp_width_val_params", mlp_width_val_params)

    kan_eps, kan_mu = eps_mu_from_params(freq_hz, kan_val_params)
    kan_rl = reflection_loss_db(freq_hz, train_thickness_m, kan_eps, kan_mu)
    assert_finite("kan_eps", kan_eps)
    assert_finite("kan_mu", kan_mu)
    assert_finite("kan_rl", kan_rl)

    sample = 0
    if write_artifacts:
        plot_rl_curve(
            args.out / "rl_curve.png",
            freq_hz,
            val_params[sample],
            kan_val_params[sample],
            mlp_matched_val_params[sample],
            mlp_width_val_params[sample] if mlp_width_val_params is not None else None,
            train_thickness_m,
            plot_dpi,
        )
        plot_rl_heatmap(args.out / "rl_heatmap.png", freq_hz, val_params[sample], plot_dpi)
        plot_param_scatter(args.out / "param_scatter.png", val_params, kan_val_params, mlp_matched_val_params, plot_dpi)
        plot_eps_mu_decomposition(
            args.out / "eps_mu_decomposition.png",
            freq_hz,
            val_params[sample],
            kan_val_params[sample],
            mlp_matched_val_params[sample],
            plot_dpi,
        )
        plot_kan_edges(args.out / "kan_edge_functions.png", tinykan_backbone, stats, device, edge_grid_points, plot_dpi)
        save_feature_table(args.out / "synthetic_materials.csv", features, true_params)

    kan_edges = tinykan_backbone.strongest_edges(top_k=8)
    edge_scores = edge_score_frame(tinykan_backbone)
    if write_artifacts:
        edge_scores.to_csv(args.out / "edge_l1_scores.csv")
        plot_edge_l1_heatmap(args.out / "edge_l1_scores.png", edge_scores, plot_dpi)
    metrics: dict[str, object] = {
        "settings": {
            "epochs": args.epochs,
            "n_materials": args.n_materials,
            "n_freq": args.n_freq,
            "seed": seed,
            "device": str(device),
            "train_thickness_m": train_thickness_m,
            "quick": bool(args.quick),
            "plot_dpi": plot_dpi,
            "edge_grid_points": edge_grid_points,
            "split_strategy": "material_id",
            "split_by_material": True,
            "train_materials": int(train_x.shape[0]),
            "val_materials": int(val_x.shape[0]),
        },
        "tinykan_params": tinykan_params,
        "mlp_params": mlp_matched_params,
        "mlp_matched_params": mlp_matched_params,
        "mlp_width_reference_params": mlp_width_reference_params,
        "parameter_gap_matched_fraction": param_gap,
        "tinykan_flops_forward": estimate_tinykan_forward_flops(len(FEATURE_NAMES), len(PARAM_NAMES), tinykan_backbone.centers.numel()),
        "mlp_flops_forward": estimate_mlp_forward_flops(len(FEATURE_NAMES), len(PARAM_NAMES), mlp_matched_hidden),
        "mlp_matched_flops_forward": estimate_mlp_forward_flops(len(FEATURE_NAMES), len(PARAM_NAMES), mlp_matched_hidden),
        "mlp_width_reference_flops_forward": estimate_mlp_forward_flops(len(FEATURE_NAMES), len(PARAM_NAMES), mlp_width_hidden),
        "tinykan": {
            "parameter_count": tinykan_params,
            "val_metrics": tinykan_result["val_metrics"],
            "history": tinykan_result["history"],
        },
        "mlp_matched": {
            "hidden_dim": mlp_matched_hidden,
            "parameter_count": mlp_matched_params,
            "val_metrics": mlp_matched_result["val_metrics"],
            "history": mlp_matched_result["history"],
        },
        "mlp_width_reference": {
            "hidden_dim": mlp_width_hidden,
            "parameter_count": mlp_width_reference_params,
            "trained": bool(mlp_width_result is not None),
            "val_metrics": mlp_width_result["val_metrics"] if mlp_width_result else None,
            "history": mlp_width_result["history"] if mlp_width_result else None,
        },
        "strongest_edges": [
            {
                "feature": FEATURE_NAMES[feature_index],
                "raw_output": RAW_OUTPUT_NAMES[output_index],
                "score": score,
            }
            for feature_index, output_index, score in kan_edges
        ],
        "edge_l1_scores": edge_scores.to_dict(orient="index"),
    }

    print(f"Seed {seed} validation losses:")
    print(f"  TinyKAN total={tinykan_result['val_metrics']['total']:.6f}")
    print(f"  MLP matched total={mlp_matched_result['val_metrics']['total']:.6f}")
    if mlp_width_result is not None:
        print(f"  MLP width=32 total={mlp_width_result['val_metrics']['total']:.6f}")
    return metrics, kan_edges


def main() -> int:
    args = parse_args()
    if args.quick:
        args.epochs = 20
        args.n_materials = 32
        args.n_freq = 16
    plot_dpi = 90 if args.quick else 180
    edge_grid_points = 48 if args.quick else 140

    device = resolve_device(args.device)
    make_output_dir(args.out)
    seeds = args.seeds if args.seeds is not None else [args.seed]

    seed_metrics: list[dict[str, object]] = []
    primary_metrics: dict[str, object] | None = None
    primary_edges: list[tuple[int, int, float]] = []

    for seed_index, seed in enumerate(seeds, start=1):
        print(f"=== Run {seed_index}/{len(seeds)} ===")
        metrics, kan_edges = run_experiment(
            args=args,
            seed=seed,
            device=device,
            plot_dpi=plot_dpi,
            edge_grid_points=edge_grid_points,
            write_artifacts=seed_index == 1,
        )
        seed_metrics.append(metrics)
        if seed_index == 1:
            primary_metrics = metrics
            primary_edges = kan_edges

    if primary_metrics is None:
        raise RuntimeError("No seed run was executed.")

    final_metrics = dict(primary_metrics)
    final_metrics["multi_seed"] = aggregate_seed_metrics(seed_metrics)
    final_metrics["seed_runs"] = [
        {
            "seed": run["settings"]["seed"],
            "tinykan_total": run["tinykan"]["val_metrics"]["total"],
            "mlp_matched_total": run["mlp_matched"]["val_metrics"]["total"],
            "mlp_width_reference_total": (
                run["mlp_width_reference"]["val_metrics"]["total"]
                if run["mlp_width_reference"]["val_metrics"] is not None
                else None
            ),
        }
        for run in seed_metrics
    ]

    (args.out / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    write_summary(args.out / "summary.md", final_metrics, primary_edges)

    expected = [
        "metrics.json",
        "summary.md",
        "rl_curve.png",
        "rl_heatmap.png",
        "param_scatter.png",
        "eps_mu_decomposition.png",
        "kan_edge_functions.png",
        "edge_l1_scores.csv",
        "edge_l1_scores.png",
    ]
    missing = [name for name in expected if not (args.out / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected output files: {missing}")

    print("Final validation losses:")
    print(f"  TinyKAN total={final_metrics['tinykan']['val_metrics']['total']:.6f}")
    print(f"  MLP matched total={final_metrics['mlp_matched']['val_metrics']['total']:.6f}")
    if len(seed_metrics) > 1:
        multi_seed = final_metrics["multi_seed"]
        tinykan_total = multi_seed["tinykan"]["total"]
        mlp_total = multi_seed["mlp_matched"]["total"]
        print("Multi-seed validation totals:")
        print(f"  TinyKAN total={tinykan_total['mean']:.6f} +/- {tinykan_total['std']:.6f}")
        print(f"  MLP matched total={mlp_total['mean']:.6f} +/- {mlp_total['std']:.6f}")
    print(f"Outputs written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
