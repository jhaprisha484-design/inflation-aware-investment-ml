"""Differential-equation models for investment growth under inflation."""
from __future__ import annotations
import numpy as np


def nominal_value(initial: float, annual_return: float, years):
    """V_nominal(t) = V0 * exp(r*t), with rates expressed as decimals."""
    t = np.asarray(years, dtype=float)
    return initial * np.exp(annual_return * t)


def real_value_constant(initial: float, annual_return: float, inflation_rate: float, years):
    """Fisher-approximated real value: V0 * exp((r-i)*t)."""
    t = np.asarray(years, dtype=float)
    return initial * np.exp((annual_return - inflation_rate) * t)


def real_value_time_varying(initial: float, annual_return: float, years, inflation_rates):
    """
    Numerically integrate dV/dt = (r - i(t))V using cumulative trapezoidal
    integration of the instantaneous real growth rate.
    """
    t = np.asarray(years, dtype=float)
    rates = np.asarray(inflation_rates, dtype=float)
    if len(t) != len(rates):
        raise ValueError("years and inflation_rates must have the same length.")
    if len(t) == 0:
        return np.array([])
    dt = np.diff(t)
    growth = annual_return - rates
    cumulative_integral = np.zeros_like(t)
    if len(t) > 1:
        cumulative_integral[1:] = np.cumsum(
            0.5 * (growth[:-1] + growth[1:]) * dt
        )
    return initial * np.exp(cumulative_integral)


def make_inflation_scenario(scenario: str, years, base_rate: float,
                            amplitude: float = 0.02, period: float = 5.0,
                            shock_year: float = 10.0, post_shock_rate: float = 0.07):
    """Return an inflation-rate array for constant, cyclical, or piecewise scenarios."""
    t = np.asarray(years, dtype=float)
    if scenario == "Constant":
        rates = np.full_like(t, base_rate, dtype=float)
    elif scenario == "Cyclical":
        rates = base_rate + amplitude * np.sin(2 * np.pi * t / period)
    elif scenario == "Piecewise":
        rates = np.where(t < shock_year, base_rate, post_shock_rate).astype(float)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return np.clip(rates, -0.99, None)
