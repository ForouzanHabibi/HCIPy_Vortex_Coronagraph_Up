"""Independent HCIPy demonstration: Zernike aberrations in a scalar vortex coronagraph.

All coordinates are normalized: pupil diameter=1, wavelength=1;
focal coordinates are in lambda/D. No measured/FDTD data are used.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def zernike_mode(x: np.ndarray, y: np.ndarray, name: str) -> np.ndarray:
    """Low-order, unit-disk Zernike modes with unit RMS over a continuous disk.

    Defocus: sqrt(3)(2r²-1); astigmatism: sqrt(6)(x²-y²);
    coma x: sqrt(8)(3r³-2r)cos(theta).
    Outside the disk values are irrelevant because of aperture masking.
    """
    r2 = x*x + y*y
    if name == 'defocus':
        return np.sqrt(3) * (2*r2 - 1)
    if name == 'astigmatism':
        return np.sqrt(6) * (x*x - y*y)
    if name == 'coma':
        return np.sqrt(8) * (3*r2 - 2) * x
    raise ValueError(f'Unknown Zernike mode: {name}')


def circular_annulus_average(intensity, x, y, inner=2., outer=4.):
    """Mean science-plane contrast in the annulus [inner, outer) lambda/D."""
    radius = np.hypot(x, y)
    mask = (radius >= inner) & (radius < outer)
    if not np.any(mask):
        raise ValueError('Focal grid does not sample requested annulus.')
    return float(np.mean(np.asarray(intensity)[mask]))


def run_simulation(output: Path, samples=128, charge=2, mode='astigmatism',
                   rms_waves=(0.0, 0.005, 0.01, 0.02, 0.04)):
    from hcipy import (make_pupil_grid, make_focal_grid, FraunhoferPropagator,
                       evaluate_supersampled, make_circular_aperture,
                       Wavefront, VortexCoronagraph, Apodizer)
    output.mkdir(parents=True, exist_ok=True)
    pupil_grid = make_pupil_grid(samples, 1.5)
    focal_grid = make_focal_grid(q=4, num_airy=8)
    propagate = FraunhoferPropagator(pupil_grid, focal_grid)
    aperture = evaluate_supersampled(make_circular_aperture(1), pupil_grid, 4)
    stop_mask = evaluate_supersampled(make_circular_aperture(0.95), pupil_grid, 4)
    vortex = VortexCoronagraph(pupil_grid, charge=charge)
    lyot_stop = Apodizer(stop_mask)

    # Reference peak: unocculted, unaberrated star, same entrance-aperture amplitude.
    reference = propagate(Wavefront(aperture)).intensity
    peak = float(np.max(reference))
    if peak <= 0:
        raise RuntimeError('Invalid reference intensity peak')
    x = np.asarray(focal_grid.x)
    y = np.asarray(focal_grid.y)

    # Zernike defined in normalized pupil coordinates (radius=1 at pupil rim).
    z_mode = zernike_mode(2*np.asarray(pupil_grid.x),
                          2*np.asarray(pupil_grid.y), mode)
    # Discrete weighted RMS correction to ensure a requested OPD RMS (waves).
    weights = np.asarray(aperture)
    mean = np.sum(weights*z_mode) / np.sum(weights)
    z_mode = z_mode - mean
    measured_rms = np.sqrt(np.sum(weights*z_mode*z_mode) / np.sum(weights))
    z_mode = z_mode / measured_rms

    cases = {}
    rows = []
    for rms in rms_waves:
        phase = 2 * np.pi * rms * z_mode
        wavefront = Wavefront(aperture * np.exp(1j*phase))
        # Scalar vortex -> Lyot stop -> final science image.
        img = propagate(lyot_stop(vortex(wavefront))).intensity
        contrast = np.asarray(img) / peak
        cases[rms] = contrast
        annular = circular_annulus_average(contrast, x, y)
        rows.append((rms, annular))
        print(f'{mode:12s} RMS OPD={rms:6.3f} waves | mean contrast (2-4 λ/D)={annular:.4e}')

    with (output/'contrast.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['rms_opd_waves', 'mean_annular_contrast_2_to_4_lambda_over_D'])
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    shown = [0.0, rms_waves[len(rms_waves)//2], rms_waves[-1]]
    for ax, rms in zip(axes, shown):
        image = np.log10(np.maximum(cases[rms].reshape(focal_grid.shape), 1e-12))
        ax.imshow(image, extent=(-8,8,-8,8), origin='lower', cmap='inferno', vmin=-9,vmax=-2)
        ax.set_title(f'{mode}: {rms:g} waves RMS')
        ax.set_xlabel('λ/D')
        ax.set_ylabel('λ/D')
    fig.savefig(output/'science_images.png', dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.3), constrained_layout=True)
    ax.plot([row[0] for row in rows], [row[1] for row in rows], 'o-')
    ax.set_yscale('log')
    ax.set_xlabel('Zernike OPD RMS (waves)')
    ax.set_ylabel('Mean normalized intensity, 2–4 λ/D')
    ax.set_title(f'Scalar charge-{charge} vortex: {mode} sensitivity')
    ax.grid(alpha=.25)
    fig.savefig(output/'contrast_vs_aberration.png', dpi=170)
    plt.close(fig)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('figures'))
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--charge', type=int, choices=[2,4], default=2)
    parser.add_argument('--mode', choices=['defocus','astigmatism','coma'], default='astigmatism')
    args = parser.parse_args()
    if args.samples < 64:
        parser.error('Use --samples >=64 for reasonable sampling.')
    run_simulation(args.output, args.samples, args.charge, args.mode)


if __name__ == '__main__':
    main()
