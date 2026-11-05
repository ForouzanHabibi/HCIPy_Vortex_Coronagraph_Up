"""Validation suite for the *scalar* charge-2 HCIPy vortex demo.

Produces independent/unocculted-reference-normalized PSFs, radial contrast curves,
aberration-mode sensitivity, and a small numerical-resolution study.
This is numerical verification, not a physical metalens or experimental validation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hcipy import Field

from coronagraph_sim import zernike_mode, circular_annulus_average


def radial_profile(image, radii, bin_edges):
    """Return azimuthal *mean* in each focal-radius bin, NaN for empty bins."""
    image, radii = np.asarray(image).ravel(), np.asarray(radii).ravel()
    result = np.full(len(bin_edges) - 1, np.nan)
    for k, (low, high) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        selection = (radii >= low) & (radii < high)
        if np.any(selection):
            result[k] = np.mean(image[selection])
    return result


def make_model(samples: int, charge: int = 2, q: int = 4):
    from hcipy import (make_pupil_grid, make_focal_grid, FraunhoferPropagator,
                       evaluate_supersampled, make_circular_aperture,
                       Wavefront, VortexCoronagraph, Apodizer)

    pupil_grid = make_pupil_grid(samples, 1.5)
    focal_grid = make_focal_grid(q=q, num_airy=8)
    propagate = FraunhoferPropagator(pupil_grid, focal_grid)
    pupil = evaluate_supersampled(make_circular_aperture(1), pupil_grid, 4)
    stop = evaluate_supersampled(make_circular_aperture(.95), pupil_grid, 4)
    vortex = VortexCoronagraph(pupil_grid, charge=charge)
    lyot = Apodizer(stop)

    # The same unocculted, unaberrated reference peak is used for all modes.
    reference = np.asarray(propagate(Wavefront(pupil)).intensity)
    reference_peak = float(reference.max())
    if not (np.isfinite(reference_peak) and reference_peak > 0):
        raise RuntimeError('Invalid unocculted reference peak')

    return dict(grid=pupil_grid, focal=focal_grid, propagate=propagate,
                pupil=pupil, stop=stop, vortex=vortex, lyot=lyot,
                Wavefront=Wavefront, peak=reference_peak,
                reference=reference / reference_peak)


def make_unit_rms_mode(model, name):
    grid, pupil = model['grid'], np.asarray(model['pupil'])
    mode = zernike_mode(2 * np.asarray(grid.x), 2 * np.asarray(grid.y), name)
    mean = np.sum(pupil * mode) / np.sum(pupil)
    mode = mode - mean
    rms = np.sqrt(np.sum(pupil * mode**2) / np.sum(pupil))
    if rms <= 0:
        raise ValueError('Degenerate Zernike mode')
    return mode / rms


def observe(model, mode=None, rms_waves=0):
    pupil = np.asarray(model['pupil'])
    if mode is None:
        phase = 0
    else:
        phase = 2 * np.pi * rms_waves * mode
    electric_field = Field(pupil * np.exp(1j * phase), model['grid'])
    wave = model['Wavefront'](electric_field)
    intensity = model['propagate'](model['lyot'](model['vortex'](wave))).intensity
    return np.asarray(intensity) / model['peak']


def plot_images(output, model, baseline, aberrated):
    ref = model['reference'].reshape(model['focal'].shape)
    images = [ref, baseline.reshape(ref.shape), aberrated.reshape(ref.shape)]
    labels = ['Without coronagraph (ideal pupil)',
              'Vortex + Lyot (zero aberration)',
              'Vortex + Lyot (0.020 waves astigmatism)']
    x, y = np.asarray(model['focal'].x), np.asarray(model['focal'].y)
    bounds = (x.min(), x.max(), y.min(), y.max())
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4), constrained_layout=True)
    for ax, im, title in zip(axes, images, labels):
        ax.imshow(np.log10(np.maximum(im, 1e-12)), origin='lower',
                  extent=bounds, vmin=-10, vmax=0, cmap='inferno')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Angular separation (lambda/D)')
        ax.set_ylabel('lambda/D')
    fig.suptitle('Shared normalization: peak of unocculted, unaberrated PSF', fontsize=11)
    fig.savefig(output / 'validation_common_scale_psfs.png', dpi=160)
    plt.close(fig)


def run(output: Path, samples=128, charge=2, convergence_samples=(96, 128, 160)):
    output.mkdir(parents=True, exist_ok=True)
    print('1/3  Shared-scale PSFs and radial contrast ...', flush=True)
    model = make_model(samples, charge)
    baseline = observe(model)
    astig = observe(model, make_unit_rms_mode(model, 'astigmatism'), .020)
    plot_images(output, model, baseline, astig)

    x, y = np.asarray(model['focal'].x), np.asarray(model['focal'].y)
    rr = np.hypot(x, y)
    bins = np.arange(0, 8.001, .25)
    centers = (bins[:-1] + bins[1:]) / 2
    curves = [('Unocculted PSF', model['reference']),
              ('Vortex, ideal', baseline),
              ('Vortex, astigmatism 0.02 waves', astig)]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, values in curves:
        profile = radial_profile(values, rr, bins)
        ax.semilogy(centers, profile, label=name)
    ax.set_ylim(1e-11, 2)
    ax.set_xlim(0, 8)
    ax.set_xlabel('Angular separation (lambda/D)')
    ax.set_ylabel('Azimuthal mean intensity / unocculted reference peak')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output / 'validation_radial_contrast.png', dpi=160)
    plt.close(fig)

    print('2/3  Defocus / astigmatism / coma sensitivity ...', flush=True)
    aberrations = ('defocus', 'astigmatism', 'coma')
    levels = (0.0, .005, .01, .02, .04)
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name in aberrations:
        z = make_unit_rms_mode(model, name)
        values = []
        for level in levels:
            image = baseline if level == 0 else observe(model, z, level)
            contrast = circular_annulus_average(image, x, y)
            rows.append((name, level, contrast))
            values.append(contrast)
            print(f'  {name:12s}  RMS={level:.3f} waves  2-4 lambda/D={contrast:.4e}', flush=True)
        ax.semilogy(levels, values, 'o-', label=name)
    ax.set_xlabel('Zernike OPD RMS (waves)')
    ax.set_ylabel('Annular mean normalized intensity (2–4 lambda/D)')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output / 'validation_zernike_comparison.png', dpi=160)
    plt.close(fig)
    with (output / 'validation_zernike_comparison.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('mode', 'rms_waves', 'mean_contrast_2_to_4_lambda_over_D'))
        writer.writerows(rows)

    print('3/3  Resolution sensitivity (fixed focal grid) ...', flush=True)
    resolution_rows = []
    for n in sorted(set(convergence_samples)):
        candidate = make_model(n, charge)
        x2, y2 = np.asarray(candidate['focal'].x), np.asarray(candidate['focal'].y)
        image0 = observe(candidate)
        image1 = observe(candidate, make_unit_rms_mode(candidate, 'astigmatism'), .01)
        value0 = circular_annulus_average(image0, x2, y2)
        value1 = circular_annulus_average(image1, x2, y2)
        resolution_rows.append((n, value0, value1))
        print(f'  samples={n:3d}: ideal={value0:.4e}, astig 0.01={value1:.4e}', flush=True)
    with (output / 'validation_convergence.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('pupil_samples', 'ideal_contrast', 'astig_0p01_contrast'))
        writer.writerows(resolution_rows)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.semilogy([row[0] for row in resolution_rows], [row[1] for row in resolution_rows],
                'o-', label='Zero aberration numerical floor')
    ax.semilogy([row[0] for row in resolution_rows], [row[2] for row in resolution_rows],
                's-', label='Astigmatism 0.01 waves')
    ax.set_xlabel('Pupil grid samples per dimension')
    ax.set_ylabel('Annular mean normalized intensity (2–4 lambda/D)')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(output / 'validation_convergence.png', dpi=160)
    plt.close(fig)
    print('Saved six validation outputs in:', output.resolve(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('figures'))
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--charge', type=int, choices=(2, 4), default=2)
    parser.add_argument('--convergence-samples', type=int, nargs='+', default=(96, 128, 160))
    args = parser.parse_args()
    if args.samples < 64 or any(n < 64 for n in args.convergence_samples):
        parser.error('All pupil resolutions must be at least 64 samples')
    run(args.output, args.samples, args.charge, args.convergence_samples)


if __name__ == '__main__':
    main()
