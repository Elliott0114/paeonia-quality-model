#!/usr/bin/env python3
"""Run the real single-portion example; no training or performance estimation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paeonia import imaging as visible
from paeonia import spectra as hsi
from paeonia.visible import VISIBLE15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "examples/SBS-01-G1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/example")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    required = ["metadata.json", "visible.png", "vendor_mean_spectrum_292.csv",
                "archived_machine_features.csv", "expected_visible15.csv",
                "expected_preprocessing_262.csv"]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing example inputs: " + ", ".join(missing))
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise SystemExit("Use a separate output directory to preserve the example inputs.")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    canonical_id = metadata["canonical_id"]

    # HPLC values and expert grade are never passed to either feature-extraction function.
    settings = argparse.Namespace(max_image_side=metadata["max_image_side"],
                                  min_component_area=metadata["min_component_area"])
    calculated = visible.process_image(data_dir / metadata["image_file"], settings)
    archived = pd.read_csv(data_dir / "archived_machine_features.csv")
    archived_values = dict(zip(archived["feature"], archived["value"], strict=True))
    machine_names = [name for name in VISIBLE15 if name.startswith("machine_")]
    if set(archived_values) != set(machine_names):
        raise ValueError("Expected the three archived machine-vision measures")
    values = [archived_values[name] if name in machine_names else calculated[name]
              for name in VISIBLE15]
    features = pd.DataFrame({
        "canonical_id": canonical_id, "feature": VISIBLE15, "value": values,
        "source": ["archived_machine_vision" if name in machine_names else "image_computed"
                   for name in VISIBLE15],
    })

    spectrum = pd.read_csv(data_dir / "vendor_mean_spectrum_292.csv")
    wavelengths = spectrum["wavelength_nm"].to_numpy(dtype=float)
    vendor = spectrum["vendor_roi_mean"].to_numpy(dtype=float)[None, :]
    if len(wavelengths) != 292 or not np.all(np.diff(wavelengths) > 0):
        raise ValueError("Expected 292 ordered vendor-export wavelengths")
    if not np.isfinite(vendor).all():
        raise ValueError("Non-finite values in example spectrum")
    factor = hsi.exposure_factor(metadata["sample_exposure_us"], metadata["white_exposure_us"])
    adjusted = vendor * factor
    indices = hsi.fixed_wavelength_indices(wavelengths, metadata["wavelength_min"],
                                          metadata["wavelength_max"])
    branches = hsi.preprocessing_branches(
        adjusted[:, indices], wavelengths[indices], window_length=metadata["sg_window"],
        polyorder=metadata["sg_polyorder"], processing_spectra=adjusted,
        processing_wavelengths=wavelengths, output_indices=indices,
    )
    result = pd.DataFrame({"wavelength_nm": wavelengths[indices],
                           **{name: array[0] for name, array in branches.items()}})

    # Reference values are checked only AFTER calculating the outputs.
    expected_features = pd.read_csv(data_dir / "expected_visible15.csv").set_index("feature")
    expected_values = expected_features.loc[VISIBLE15, "value"].to_numpy(dtype=float)
    # Image arithmetic uses float32; reduction order can differ across NumPy builds.
    # Stored machine values are copied directly and must still match exactly.
    image_rows = features["source"].eq("image_computed").to_numpy()
    image_rtol = 8 * float(np.finfo(np.float32).eps)
    image_atol = float(np.finfo(np.float32).eps)
    np.testing.assert_allclose(features.loc[image_rows, "value"], expected_values[image_rows],
                               rtol=image_rtol, atol=image_atol)
    np.testing.assert_array_equal(features.loc[~image_rows, "value"], expected_values[~image_rows])
    expected_spectra = pd.read_csv(data_dir / "expected_preprocessing_262.csv")
    np.testing.assert_allclose(result["wavelength_nm"], expected_spectra["wavelength_nm"],
                               rtol=0, atol=1e-5)
    errors = {}
    for name in branches:
        # The reference CSV uses rounded decimal values.
        np.testing.assert_allclose(result[name], expected_spectra[name], rtol=1e-8, atol=1e-9)
        errors[name] = float(np.max(np.abs(result[name] - expected_spectra[name])))
    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_dir / "visible_features.csv", index=False, encoding="utf-8-sig")
    result.to_csv(output_dir / "preprocessing.csv", index=False, encoding="utf-8-sig")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), constrained_layout=True)
    labels = {"P0": "Exposure-adjusted vendor signal", "P1": "SNV",
              "P2": "SG + SNV", "P3": "SG derivative + SNV"}
    for axis, name in zip(axes.flat, branches, strict=True):
        axis.plot(result.wavelength_nm, result[name], color="#246979", linewidth=1.2)
        axis.set(title=f"{name}: {labels[name]}", xlabel="Wavelength (nm)",
                 ylabel="Vendor signal" if name == "P0" else "Standardized signal")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{canonical_id} — single-portion processing example")
    fig.savefig(output_dir / "spectral_preprocessing.png", dpi=150)
    plt.close(fig)
    report = {
        "status": "PASS", "canonical_id": canonical_id,
        "image_computed_features": 12, "archived_machine_vision_features": 3,
        "exported_wavelengths": len(wavelengths), "analysis_wavelengths": len(indices),
        "visible_max_absolute_error": float(np.max(np.abs(features["value"] - expected_values))),
        "spectral_max_absolute_errors": errors,
        "feature_tolerance": {"image_rtol": image_rtol, "image_atol": image_atol,
                              "stored_machine_values": "exact"},
        "spectral_tolerance": {"rtol": 1e-8, "atol": 1e-9},
        "scope": "Single-portion feature extraction and spectral processing only; no fitted prediction model or manuscript performance estimate.",
    }
    (output_dir / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
