"""Compare scalar vortex charges 2, 4, and 6 with HCIPy.

Independent numerical model: circular entrance pupil, scalar focal-plane vortex,
95%-diameter Lyot stop, monochromatic Fraunhofer science plane. The study reports
(i) on-axis stellar leakage in a 2-4 lambda/D annulus under low-order Zernike OPD,
and (ii) off-axis point-source *relative core throughput* (encircled intensity
inside 0.7 lambda/D around the reference source peak). No metasurface, polarimetric
Jones response, FDTD data, or measured planet contrast is represented.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def zernike_xy(x, y, name):
    """Unnormalized low-order modes over unit-radius disk; RMS normalized later."""
    r2 = x*x + y*y
    if name == 'defocus':
        return np.sqrt(3) * (2*r2 - 1)
    if name == 'astigmatism':
        return np.sqrt(6) * (x*x-y*y)
    if name == 'coma':
        return np.sqrt(8) * (3*r2-2) * x
    raise ValueError(name)


def unit_rms_mode(grid, aperture, name):
    x, y = 2*np.asarray(grid.x), 2*np.asarray(grid.y)
    z = zernike_xy(x, y, name)
    w = np.asarray(aperture)
    z -= np.sum(w*z)/np.sum(w)
    rms = np.sqrt(np.sum(w*z*z)/np.sum(w))
    if rms <= 0 or not np.isfinite(rms):
        raise ValueError(f'Invalid RMS for mode {name}')
    return z/rms


def annular_mean(image, x, y, inner=2., outer=4.):
    mask = (np.hypot(x,y)>=inner)&(np.hypot(x,y)<outer)
    if not np.any(mask):
        raise ValueError('Empty annulus')
    return float(np.mean(np.asarray(image)[mask]))


def core_fraction(img, x, y, x0, y0=0., aperture_radius=0.7):
    """Pixel-area factor cancels in the throughput ratio with a fixed focal grid."""
    inside = (x-x0)**2 + (y-y0)**2 <= aperture_radius**2
    if not np.any(inside):
        raise ValueError('Empty photometric aperture')
    return float(np.sum(np.asarray(img)[inside]))


def make_system(samples, charge, q=4, num_airy=9):
    from hcipy import (make_pupil_grid, make_focal_grid, make_circular_aperture,
                       evaluate_supersampled, FraunhoferPropagator,
                       VortexCoronagraph, Apodizer)
    pupil_grid = make_pupil_grid(samples, 1.5)
    focal_grid = make_focal_grid(q=q, num_airy=num_airy)
    pupil = evaluate_supersampled(make_circular_aperture(1), pupil_grid, 4)
    lyot_mask = evaluate_supersampled(make_circular_aperture(.95), pupil_grid, 4)
    return {
        'grid': pupil_grid, 'focal': focal_grid, 'pupil': pupil,
        'propagator': FraunhoferPropagator(pupil_grid, focal_grid),
        'vortex': VortexCoronagraph(pupil_grid, charge=charge),
        'lyot': Apodizer(lyot_mask),
    }


def image_of(system, amplitude, with_coronagraph=True):
    from hcipy import Field, Wavefront
    field = Field(np.asarray(amplitude, dtype=complex).ravel(), system['grid'])
    wave = Wavefront(field)
    if with_coronagraph:
        wave = system['lyot'](system['vortex'](wave))
    else:
        # Include the same Lyot stop in planet reference: ratio isolates mask throughput.
        wave = system['lyot'](wave)
    return np.asarray(system['propagator'](wave).intensity).ravel()


def run(output, samples, charges, rms_values, separations, mode, photometry_radius):
    output.mkdir(parents=True, exist_ok=True)
    leakage_rows, throughput_rows = [], []
    fig_l, ax_l = plt.subplots(figsize=(7.2,4.8), constrained_layout=True)
    fig_t, ax_t = plt.subplots(figsize=(7.2,4.8), constrained_layout=True)
    fig_i, axs = plt.subplots(1,len(charges),figsize=(4.1*len(charges),4.1),
                           constrained_layout=True, squeeze=False)

    for col, charge in enumerate(charges):
        print(f'Charge {charge}: constructing coronagraph ...', flush=True)
        s = make_system(samples, charge)
        grid = s['grid']
        x, y = np.asarray(s['focal'].x), np.asarray(s['focal'].y)
        pupil = np.asarray(s['pupil'])
        z = unit_rms_mode(grid, pupil, mode)
        star_reference = image_of(s, pupil, with_coronagraph=False)
        # All leakage values normalized to *unocculted, no-Lyot* peak, same pupil.
        from hcipy import Field, Wavefront
        raw_ref = np.asarray(s['propagator'](
            Wavefront(Field(pupil.astype(complex),grid))).intensity)
        star_peak = float(np.max(raw_ref))
        if star_peak<=0:
            raise RuntimeError('Invalid stellar peak')
        leaks=[]
        for rms in rms_values:
            phase = 2*np.pi*rms*z
            star = image_of(s, pupil*np.exp(1j*phase), True)
            leakage = annular_mean(star/star_peak, x,y)
            leaks.append(leakage)
            leakage_rows.append((charge,mode,rms,leakage))
            print(f'  stellar {mode}={rms:.3f} waves: {leakage:.5e}',flush=True)
        ax_l.semilogy(rms_values,leaks,'o-',label=f'charge {charge}')

        reference = image_of(s,pupil,True) / star_peak
        axs[0,col].imshow(np.log10(np.maximum(reference.reshape(s['focal'].shape),1e-12)),
                          origin='lower',extent=(x.min(),x.max(),y.min(),y.max()),
                          cmap='inferno',vmin=-10,vmax=-3)
        axs[0,col].set_title(f'Charge {charge}: ideal star leakage')
        axs[0,col].set_xlabel('lambda/D')
        axs[0,col].set_ylabel('lambda/D')
        axs[0,col].set_xlim(-6,6)
        axs[0,col].set_ylim(-6,6)

        throughput=[]
        for sep in separations:
            # Off-axis source at angular separation sep*(lambda/D) along x.
            # Pupil diameter is 1 and lambda=1 in the normalized model.
            tilt=np.exp(2j*np.pi*sep*np.asarray(grid.x))
            incident=pupil*tilt
            ref=image_of(s,incident,with_coronagraph=False)
            coro=image_of(s,incident,with_coronagraph=True)
            # Measure both in SAME aperture, centered on reference PSF peak;
            # this avoids needing to assume Fourier-propagator sign convention.
            idx=int(np.argmax(ref))
            x0,y0=float(x[idx]),float(y[idx])
            if abs(abs(x0)-sep)>0.65 or abs(y0)>0.65:
                raise RuntimeError(f'Off-axis source peak unexpected: sep={sep}, peak=({x0},{y0})')
            denominator=core_fraction(ref,x,y,x0,y0,photometry_radius)
            numerator=core_fraction(coro,x,y,x0,y0,photometry_radius)
            if denominator<=0:
                raise RuntimeError('Non-positive off-axis reference core energy')
            ratio=numerator/denominator
            throughput_rows.append((charge,sep,ratio,numerator,denominator))
            throughput.append(ratio)
            print(f'  off-axis {sep:.1f} lambda/D: relative core throughput={ratio:.5f}',flush=True)
        ax_t.plot(separations,throughput,'o-',label=f'charge {charge}')

    ax_l.set(xlabel='Zernike OPD RMS (waves)',
             ylabel='Mean stellar intensity / unocculted reference peak (2–4 lambda/D)',
             title=f'Scalar vortex charge comparison: {mode}')
    ax_l.grid(alpha=.3)
    ax_l.legend()
    fig_l.savefig(output/'charge_stellar_leakage.png',dpi=170)
    plt.close(fig_l)

    ax_t.set(xlabel='Off-axis separation (lambda/D)',
             ylabel=f'Core throughput relative to Lyot-only reference (r={photometry_radius:g} lambda/D)',
             title='Off-axis response and charge trade-off')
    ax_t.set_ylim(bottom=0)
    ax_t.grid(alpha=.3)
    ax_t.legend()
    fig_t.savefig(output/'charge_offaxis_throughput.png',dpi=170)
    plt.close(fig_t)

    fig_i.suptitle('On-axis point-source residual intensity (shared log scale)')
    fig_i.savefig(output/'charge_ideal_stellar_images.png',dpi=170)
    plt.close(fig_i)

    with (output/'charge_stellar_leakage.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(('charge','mode','rms_waves','mean_stellar_intensity_2_to_4_lambda_over_D'));writer.writerows(leakage_rows)
    with (output/'charge_offaxis_throughput.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.writer(f);writer.writerow(('charge','separation_lambda_over_D','relative_core_throughput','coronagraph_core_sum','lyot_only_core_sum'));writer.writerows(throughput_rows)
    print('Saved 3 PNG figures and 2 CSV tables to:',output.resolve(),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('figures'))
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--charges',type=int,nargs='+',default=[2,4,6])
    p.add_argument('--mode',choices=('defocus','astigmatism','coma'),default='astigmatism')
    p.add_argument('--photometry-radius',type=float,default=.7)
    a=p.parse_args()
    if a.samples<64 or a.samples>512:p.error('samples must be 64..512')
    if len(set(a.charges))!=len(a.charges) or any(c<=0 or c%2 for c in a.charges):p.error('charges must be unique positive even integers')
    if not 0<a.photometry_radius<2:p.error('photometry radius must be between 0 and 2 lambda/D')
    run(a.output,a.samples,a.charges,(0.,.005,.01,.02,.04),
        (1.,1.5,2.,2.5,3.,4.,5.,6.),a.mode,a.photometry_radius)

if __name__=='__main__':
    main()
