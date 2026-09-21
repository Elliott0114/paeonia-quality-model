"""Exposure rescaling and the P0–P3 spectral representations."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def exposure_factor(sample_exposure_us: float, white_exposure_us: float) -> float:
    if not np.isfinite(sample_exposure_us) or sample_exposure_us <= 0:
        raise ValueError("sample exposure must be positive and finite")
    if not np.isfinite(white_exposure_us) or white_exposure_us <= 0:
        raise ValueError("white exposure must be positive and finite")
    return float(white_exposure_us / sample_exposure_us)


def spectral_snv(spectra: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float64)
    means = spectra.mean(axis=-1, keepdims=True)
    stds = spectra.std(axis=-1, ddof=0, keepdims=True)
    if np.any(~np.isfinite(stds)) or np.any(stds <= eps):
        raise ValueError("SNV encountered a constant or non-finite spectrum")
    return (spectra - means) / stds


def preprocessing_branches(
    spectra: np.ndarray,
    wavelengths: np.ndarray,
    window_length: int = 11,
    polyorder: int = 2,
    *,
    processing_spectra: np.ndarray | None = None,
    processing_wavelengths: np.ndarray | None = None,
    output_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    spectra = np.asarray(spectra, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    if spectra.ndim != 2 or spectra.shape[1] != wavelengths.size:
        raise ValueError("spectra/wavelength shape mismatch")
    if window_length % 2 != 1 or window_length <= polyorder or window_length > wavelengths.size:
        raise ValueError("invalid Savitzky-Golay settings")
    if processing_spectra is None:
        processing_spectra = spectra
        processing_wavelengths = wavelengths
        output_indices = np.arange(wavelengths.size)
    if processing_wavelengths is None or output_indices is None:
        raise ValueError("processing wavelengths and output indices must be supplied together")
    processing_spectra = np.asarray(processing_spectra, dtype=np.float64)
    processing_wavelengths = np.asarray(processing_wavelengths, dtype=np.float64)
    output_indices = np.asarray(output_indices, dtype=int)
    if processing_spectra.shape[1] != processing_wavelengths.size:
        raise ValueError("processing spectra/wavelength shape mismatch")
    if not np.array_equal(processing_wavelengths[output_indices], wavelengths):
        raise ValueError("processing output indices do not reproduce requested wavelengths")
    smooth_full = savgol_filter(
        processing_spectra,
        window_length=window_length,
        polyorder=polyorder,
        deriv=0,
        axis=1,
        mode="interp",
    )
    derivative_full = np.gradient(smooth_full, processing_wavelengths, axis=1, edge_order=2)
    smooth = smooth_full[:, output_indices]
    derivative = derivative_full[:, output_indices]
    return {
        "P0": spectra.copy(),
        "P1": spectral_snv(spectra),
        "P2": spectral_snv(smooth),
        "P3": spectral_snv(derivative),
    }


def fixed_wavelength_indices(wavelengths: np.ndarray, low: float, high: float) -> np.ndarray:
    idx = np.flatnonzero((wavelengths >= low) & (wavelengths <= high))
    if idx.size == 0:
        raise ValueError(f"No wavelength in [{low}, {high}]")
    if not np.all(np.diff(idx) == 1):
        raise ValueError("Selected wavelength range is not contiguous")
    return idx
