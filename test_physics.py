from __future__ import annotations

import math
import unittest

import torch

try:
    from physics import C0, cole_cole_polar, reflection_loss_db
except ImportError:  # pragma: no cover
    from .physics import C0, cole_cole_polar, reflection_loss_db


class PhysicsLayerTest(unittest.TestCase):
    def test_cole_cole_kk_consistent(self) -> None:
        freq = torch.logspace(5, 15, 5000)
        eps_inf = torch.tensor([2.3])
        eps_s = torch.tensor([7.9])
        tau = torch.tensor([1.2e-10])
        alpha = torch.tensor([0.0])
        eps = cole_cole_polar(freq, eps_inf, eps_s, tau, alpha).squeeze(0)

        for target_hz in [2.0e9, 8.0e9, 16.0e9]:
            target = torch.tensor(target_hz)
            mask = torch.abs(torch.log(freq / target)) > 0.015
            xi = freq[mask]
            eps_imag = eps.imag[mask]
            integrand = xi * eps_imag / (xi.pow(2) - target.pow(2))
            kk_real = eps_inf.item() - (2.0 / math.pi) * torch.trapezoid(integrand, xi).item()
            direct_real = cole_cole_polar(target[None], eps_inf, eps_s, tau, alpha).real.item()
            self.assertLess(abs(kk_real - direct_real) / direct_real, 0.08)

    def test_rl_known_case(self) -> None:
        freq = torch.tensor([10.0e9])
        matched_lossy = torch.tensor([[1.0 - 3.0j]], dtype=torch.complex64)
        rl = reflection_loss_db(freq, 0.05, matched_lossy, matched_lossy)
        self.assertLess(float(rl.item()), -35.0)

    def test_rl_quarter_wave(self) -> None:
        freq = torch.tensor([10.0e9])
        eps_r = torch.tensor([4.0 - 0.8j], dtype=torch.complex64)
        mu_r = torch.tensor([1.0 - 0.15j], dtype=torch.complex64)
        thicknesses = torch.linspace(1.0e-3, 7.0e-3, 700)
        rl = reflection_loss_db(freq, thicknesses, eps_r, mu_r).squeeze(1)

        observed = thicknesses[torch.argmin(rl)].item()
        expected = C0 / (4.0 * freq.item() * math.sqrt(abs((eps_r * mu_r).item())))
        self.assertLess(abs(observed - expected) / expected, 0.35)


if __name__ == "__main__":
    unittest.main()
