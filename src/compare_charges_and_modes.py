"""Scalar vortex charge x aberration study using HCIPy.

Place this script beside compare_vortex_charges.py. Reuses the previously
validated model and a consistent unocculted reference PSF normalization.
This is a numerical sensitivity study, NOT a vector-vortex/metasurface model,
physical detection SNR, or laboratory validation.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_vortex_charges import (
    make_system, unit_rms_mode, image_of, annular_mean, core_fraction
)

MODES = ('defocus', 'astigmatism', 'coma')
LEVELS = (0.0, 0.005, 0.010, 0.020, 0.040)
CHARGES = (2, 4, 6)


def analyse(system, charge, rms_levels=LEVELS):
    """Return leakage for all modes with one shared per-grid reference peak."""
    from hcipy import Field, Wavefront

    grid = system['grid']
    pupil = np.asarray(system['pupil'])
    focal = system['focal']
    x, y = np.asarray(focal.x), np.asarray(focal.y)
    reference = system['propagator'](Wavefront(Field(pupil.astype(complex), grid)))
    peak = float(np.max(np.asarray(reference.intensity)))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError('Invalid reference PSF peak')

    baseline = annular_mean(image_of(system, pupil, True) / peak, x, y)
    rows = []
    for mode in MODES:
        zernike = unit_rms_mode(grid, pupil, mode)
        for level in rms_levels:
            if level == 0:
                leakage = baseline
            else:
                incident = pupil * np.exp(2j * np.pi * level * zernike)
                stellar_psf = image_of(system, incident, True)
                leakage = annular_mean(stellar_psf / peak, x, y)
            rows.append((charge, mode, level, leakage))
            print(f'  charge={charge} {mode:12s} RMS={level:.3f}: {leakage:.5e}', flush=True)
    return rows


def offaxis_core_throughput(system, separation=2.0, radius=0.7):
    """Relative core throughput vs Lyot-only, not total transmitted power."""
    grid = system['grid']
    pupil = np.asarray(system['pupil'])
    x = np.asarray(system['focal'].x)
    y = np.asarray(system['focal'].y)
    incident = pupil * np.exp(2j * np.pi * separation * np.asarray(grid.x))
    ref = image_of(system, incident, False)
    coro = image_of(system, incident, True)
    peak_index = int(np.argmax(ref))
    x0, y0 = float(x[peak_index]), float(y[peak_index])
    if abs(abs(x0) - separation) > 0.65 or abs(y0) > 0.65:
        raise RuntimeError('Unexpected off-axis reference source location')
    denominator = core_fraction(ref, x, y, x0, y0, radius)
    if denominator <= 0:
        raise ValueError('Zero Lyot-only core reference')
    return core_fraction(coro, x, y, x0, y0, radius) / denominator


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('figures'))
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--rms', type=float, default=0.020,
                        help='RMS OPD in waves for heatmap and illustrative ratio')
    parser.add_argument('--separation', type=float, default=2.0,
                        help='Off-axis source separation in lambda/D')
    parser.add_argument('--convergence-samples', type=int, nargs='+',
                        default=(96, 128, 160))
    args = parser.parse_args()
    if args.samples < 64 or any(n < 64 for n in args.convergence_samples):
        parser.error('All pupil sample counts must be >= 64')
    if args.rms <= 0 or args.separation <= 0:
        parser.error('RMS and separation must be positive')
    args.output.mkdir(exist_ok=True, parents=True)

    rows = []
    ratios = []
    print('1/3 Charge x Zernike mode sensitivity...', flush=True)
    for charge in CHARGES:
        system = make_system(args.samples, charge)
        # Include the requested heatmap level, even if not one of the sweep levels.
        levels = tuple(sorted(set(LEVELS + (args.rms,))))
        rows.extend(analyse(system, charge, levels))
        throughput = offaxis_core_throughput(system, args.separation)
        for mode in MODES:
            leakage = next(r[3] for r in rows if r[0] == charge and r[1] == mode
                           and abs(r[2] - args.rms) < 1e-12)
            # Diagnostic ratio ONLY, not SNR or a physical planet-detection metric.
            ratios.append((charge, mode, args.rms, args.separation,
                           throughput, leakage, throughput / leakage if leakage > 0 else np.nan))
        print(f'  off-axis relative core throughput charge {charge}: {throughput:.5f}', flush=True)

    write_csv(args.output / 'charge_mode_leakage.csv',
              ('charge', 'mode', 'rms_waves', 'stellar_annular_mean_2_4_lambda_over_D'), rows)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), sharey=True,
                             constrained_layout=True)
    for ax, mode in zip(axes, MODES):
        for charge in CHARGES:
            subset = [r for r in rows if r[0] == charge and r[1] == mode]
            ax.semilogy([r[2] for r in subset], [r[3] for r in subset], 'o-',
                        label=f'charge {charge}')
        ax.set_title(mode.title())
        ax.set_xlabel('RMS OPD (waves)')
        ax.grid(alpha=.25)
    axes[0].set_ylabel('Annular stellar intensity / unocculted PSF peak')
    axes[-1].legend()
    fig.suptitle('Scalar vortex: charge × aberration sensitivity (2–4 λ/D)')
    fig.savefig(args.output / 'charge_mode_leakage.png', dpi=170)
    plt.close(fig)

    matrix = np.array([[next(r[3] for r in rows if r[0] == charge and r[1] == mode
                             and abs(r[2] - args.rms) < 1e-12) for mode in MODES]
                       for charge in CHARGES])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    mappable = ax.imshow(np.log10(np.maximum(matrix, 1e-15)), cmap='viridis', aspect='auto')
    ax.set_xticks(range(3), [m.title() for m in MODES])
    ax.set_yticks(range(3), [f'Charge {c}' for c in CHARGES])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:.2e}', ha='center', va='center',
                    color='white' if np.log10(max(matrix[i,j], 1e-15)) < -5 else 'black', fontsize=9)
    fig.colorbar(mappable, ax=ax, label='log10(normalized annular stellar intensity)')
    ax.set_title(f'Stellar leakage at RMS = {args.rms:g} waves')
    fig.savefig(args.output / 'charge_mode_heatmap_rms_0p02.png', dpi=170)
    plt.close(fig)

    write_csv(args.output / 'charge_tradeoff_score.csv',
              ('charge', 'mode', 'rms_waves', 'separation_lambda_over_D',
               'relative_core_throughput', 'annular_stellar_leakage',
               'illustrative_throughput_over_annular_leakage'), ratios)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for mode in MODES:
        values = [next(r[-1] for r in ratios if r[0] == charge and r[1] == mode)
                  for charge in CHARGES]
        ax.semilogy(CHARGES, values, 'o-', label=mode)
    ax.set_xticks(CHARGES)
    ax.set_xlabel('Topological charge')
    ax.set_ylabel('Relative core throughput / annular stellar leakage')
    ax.set_title('Illustrative ratio only — NOT detection SNR')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(args.output / 'charge_tradeoff_score.png', dpi=170)
    plt.close(fig)

    print('2/3 Charge 6 resolution sensitivity...', flush=True)
    convergence = []
    for n in sorted(set(args.convergence_samples)):
        system = make_system(n, 6)
        results = analyse(system, 6, (0.0, args.rms))
        for mode in MODES:
            ideal = next(r[3] for r in results if r[1] == mode and r[2] == 0.0)
            aberrated = next(r[3] for r in results if r[1] == mode and r[2] == args.rms)
            convergence.append((n, mode, ideal, aberrated))
    write_csv(args.output / 'charge6_convergence.csv',
              ('pupil_samples', 'mode', 'zero_opd_leakage', 'aberrated_leakage'), convergence)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    resolutions = sorted(set(args.convergence_samples))
    axes[0].semilogy(resolutions,
                     [next(r[2] for r in convergence if r[0] == n) for n in resolutions],
                     'o-', label='Zero OPD')
    for mode in MODES:
        values = [next(r[3] for r in convergence if r[0] == n and r[1] == mode)
                  for n in resolutions]
        axes[1].semilogy(resolutions, values, 'o-', label=mode)
    for ax in axes:
        ax.set_xlabel('Pupil samples per dimension')
        ax.set_ylabel('Normalized stellar leakage (2–4 λ/D)')
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_title('Charge 6: zero-aberration numerical floor')
    axes[1].set_title(f'Charge 6: {args.rms:g} waves RMS')
    fig.savefig(args.output / 'charge6_convergence.png', dpi=170)
    plt.close(fig)
    print('3/3 Saved 4 figures and 3 CSVs to:', args.output.resolve(), flush=True)
    print('Caution: illustrative ratio uses annular stellar leakage, not local planet noise.', flush=True)


if __name__ == '__main__':
    main()
