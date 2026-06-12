"""Calibration: fit Heston parameters to an observed IV surface.

Two engines share one objective (IV residuals on the lattice):
- SurrogateCalibrator: residuals through the neural surrogate. One residual
  evaluation is a single forward pass (~0.1 ms), and the Jacobian can come
  from torch autograd instead of finite differences.
- direct_calibrate: residuals through the Carr-Madan FFT pricer + IV
  inversion — the classical route, kept as the benchmark.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import least_squares

from .lhs import PARAM_BOUNDS
from ..pricing.heston import PARAM_NAMES, HestonParams, feller_condition, iv_grid
from .mlp import MLPSurrogate, Normalizer

_DEFAULT_X0 = np.array([2.0, 0.06, 0.4, -0.5, 0.06])


def _bounds_arrays() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES])
    return lower, upper


@dataclass
class CalibrationResult:
    params: HestonParams
    rmse_iv: float
    n_evals: int
    wall_time_s: float
    feller_margin: float
    feller_satisfied: bool
    success: bool

    def as_dict(self) -> dict:
        return {
            **{k: getattr(self.params, k) for k in PARAM_NAMES},
            "rmse_iv": self.rmse_iv,
            "n_evals": self.n_evals,
            "wall_time_s": self.wall_time_s,
            "feller_margin": self.feller_margin,
            "feller_satisfied": self.feller_satisfied,
            "success": self.success,
        }


class SurrogateCalibrator:
    """Least-squares calibration with the neural surrogate as the pricer."""

    def __init__(
        self,
        model: MLPSurrogate,
        normalizer: Normalizer,
        moneyness: np.ndarray,
        maturities: np.ndarray,
    ) -> None:
        self.model = model.eval()
        self.normalizer = normalizer
        self.moneyness = np.asarray(moneyness)
        self.maturities = np.asarray(maturities)
        self.grid_shape = (len(self.maturities), len(self.moneyness))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Surrogate IV grid (n_mat, n_mon) for one parameter vector."""
        xn = torch.as_tensor(self.normalizer.norm_x(x[None, :]), dtype=torch.float32)
        with torch.no_grad():
            yn = self.model(xn).numpy()[0]
        return self.normalizer.denorm_y(yn).reshape(self.grid_shape)

    def _residuals(self, x: np.ndarray, market_iv: np.ndarray) -> np.ndarray:
        return (self.predict(x) - market_iv).ravel()

    def _torch_jac(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of the flattened IV prediction wrt raw params via autograd.

        Chain rule through the normalisations: d(iv)/d(x) =
        diag(y_std) . J_net . diag(1/x_std).
        """
        xn = torch.as_tensor(
            self.normalizer.norm_x(x[None, :]), dtype=torch.float32
        ).requires_grad_(True)
        yn = self.model(xn)[0]
        J = torch.autograd.functional.jacobian(
            lambda z: self.model(z)[0], xn.detach(), vectorize=True
        )[:, 0, :].numpy()
        del yn
        return (self.normalizer.y_std[:, None] * J) / self.normalizer.x_std[None, :]

    def calibrate(
        self,
        market_iv: np.ndarray,
        x0: np.ndarray | None = None,
        jac: str = "torch",
        max_nfev: int = 200,
    ) -> CalibrationResult:
        """Fit the five parameters to a (n_mat, n_mon) market IV grid.

        jac='torch' uses the autograd Jacobian (exact for the surrogate);
        jac='fd' falls back to scipy's 2-point finite differences.
        """
        market_iv = np.asarray(market_iv).reshape(self.grid_shape)
        lower, upper = _bounds_arrays()
        t0 = time.perf_counter()
        result = least_squares(
            self._residuals,
            x0 if x0 is not None else _DEFAULT_X0,
            args=(market_iv,),
            jac=(lambda x, *a: self._torch_jac(x)) if jac == "torch" else "2-point",
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-12,
            ftol=1e-12,
            # The network runs in float32: scipy's default ~1e-8 FD step sits
            # below its noise floor and zeroes out weak directions (kappa).
            diff_step=None if jac == "torch" else 1e-3,
        )
        wall = time.perf_counter() - t0
        params = HestonParams.from_array(result.x)
        margin, ok = feller_condition(params)
        return CalibrationResult(
            params=params,
            rmse_iv=float(np.sqrt(np.mean(result.fun**2))),
            n_evals=int(result.nfev),
            wall_time_s=wall,
            feller_margin=margin,
            feller_satisfied=ok,
            success=bool(result.success),
        )


def direct_calibrate(
    market_iv: np.ndarray,
    moneyness: np.ndarray,
    maturities: np.ndarray,
    x0: np.ndarray | None = None,
    max_nfev: int = 200,
    fft_n: int = 2048,
) -> CalibrationResult:
    """Classical calibration: every residual evaluation re-prices the lattice
    with one FFT per maturity plus implied-vol inversion. The benchmark the
    surrogate is measured against."""
    market_iv = np.asarray(market_iv)
    lower, upper = _bounds_arrays()

    def residuals(x: np.ndarray) -> np.ndarray:
        grid = iv_grid(HestonParams.from_array(x), moneyness, maturities, N=fft_n)
        return np.nan_to_num(grid - market_iv, nan=0.5).ravel()

    t0 = time.perf_counter()
    result = least_squares(
        residuals,
        x0 if x0 is not None else _DEFAULT_X0,
        bounds=(lower, upper),
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
    )
    wall = time.perf_counter() - t0
    params = HestonParams.from_array(result.x)
    margin, ok = feller_condition(params)
    return CalibrationResult(
        params=params,
        rmse_iv=float(np.sqrt(np.mean(result.fun**2))),
        n_evals=int(result.nfev),
        wall_time_s=wall,
        feller_margin=margin,
        feller_satisfied=ok,
        success=bool(result.success),
    )


def benchmark_calibration(
    calibrator: SurrogateCalibrator,
    true_params: HestonParams | None = None,
    x0: np.ndarray | None = None,
    n_runs: int = 3,
) -> dict:
    """Wall-time benchmark: surrogate vs direct FFT calibration.

    Both engines fit the same synthetic market surface (generated by the *FFT*
    pricer from known true parameters, so the surrogate cannot cheat) from the
    same starting point with the same optimiser settings. Returns timings,
    speed-up factor, and parameter-recovery errors.
    """
    true_params = true_params or HestonParams(kappa=3.0, theta=0.05, xi=0.45, rho=-0.65, v0=0.07)
    market_iv = iv_grid(true_params, calibrator.moneyness, calibrator.maturities)

    sur_times, dir_times = [], []
    sur_res = dir_res = None
    for _ in range(n_runs):
        sur_res = calibrator.calibrate(market_iv, x0=x0)
        sur_times.append(sur_res.wall_time_s)
    dir_res = direct_calibrate(market_iv, calibrator.moneyness, calibrator.maturities, x0=x0)
    dir_times.append(dir_res.wall_time_s)

    truth = true_params.as_array()
    return {
        "true_params": dict(zip(PARAM_NAMES, truth)),
        "surrogate": {
            "wall_time_s": float(np.median(sur_times)),
            "rmse_iv": sur_res.rmse_iv,
            "n_evals": sur_res.n_evals,
            "params": dict(zip(PARAM_NAMES, sur_res.params.as_array())),
            "param_abs_err": dict(zip(PARAM_NAMES, np.abs(sur_res.params.as_array() - truth))),
        },
        "direct": {
            "wall_time_s": float(np.median(dir_times)),
            "rmse_iv": dir_res.rmse_iv,
            "n_evals": dir_res.n_evals,
            "params": dict(zip(PARAM_NAMES, dir_res.params.as_array())),
            "param_abs_err": dict(zip(PARAM_NAMES, np.abs(dir_res.params.as_array() - truth))),
        },
        "speedup": float(np.median(dir_times) / np.median(sur_times)),
    }
