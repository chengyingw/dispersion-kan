from __future__ import annotations

import math

import torch


C0 = 3.0e8
Z0 = 377.0
EPS = 1.0e-8


def cole_cole_polar(
    freq_hz: torch.Tensor,
    inf: torch.Tensor,
    static: torch.Tensor,
    tau: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Stable Cole-Cole dispersion without complex fractional powers."""
    omega_tau = torch.clamp(2.0 * math.pi * freq_hz[None, :] * tau[:, None], min=1.0e-18)
    exponent = 1.0 - alpha[:, None]
    magnitude = torch.exp(exponent * torch.log(omega_tau))
    phase = 0.5 * math.pi * exponent
    denom = 1.0 + torch.complex(magnitude * torch.cos(phase), magnitude * torch.sin(phase))
    return inf[:, None].to(torch.complex64) + (static - inf)[:, None].to(torch.complex64) / denom


def eps_mu_from_params(freq_hz: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eps_inf, eps_s, tau_eps, alpha_eps, mu_inf, mu_s, tau_mu, alpha_mu = params.unbind(dim=-1)
    eps_r = cole_cole_polar(freq_hz, eps_inf, eps_s, tau_eps, alpha_eps)
    mu_r = cole_cole_polar(freq_hz, mu_inf, mu_s, tau_mu, alpha_mu)
    return eps_r, mu_r


def reflection_loss_db(
    freq_hz: torch.Tensor,
    thickness_m: torch.Tensor | float,
    eps_r: torch.Tensor,
    mu_r: torch.Tensor,
) -> torch.Tensor:
    """Metal-backed single-layer reflection loss in dB."""
    thickness = torch.as_tensor(thickness_m, dtype=freq_hz.dtype, device=freq_hz.device)
    if thickness.ndim == 0:
        omega_term = 2.0 * math.pi * freq_hz[None, :] * thickness / C0
    else:
        omega_term = 2.0 * math.pi * freq_hz[None, :] * thickness[:, None] / C0
        eps_r = eps_r[None, :]
        mu_r = mu_r[None, :]

    propagation = torch.sqrt(mu_r * eps_r)
    zin = Z0 * torch.sqrt(mu_r / eps_r) * torch.tanh(1j * omega_term * propagation)
    gamma = (zin - Z0) / (zin + Z0 + EPS)
    rl = 20.0 * torch.log10(torch.abs(gamma) + 1.0e-12)
    return torch.clamp(rl.real, min=-60.0, max=0.0)
