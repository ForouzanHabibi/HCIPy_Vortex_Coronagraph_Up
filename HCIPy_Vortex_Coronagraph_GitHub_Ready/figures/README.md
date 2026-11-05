# Simulation figures and data

This folder contains saved PNG figures and CSV outputs from the scalar HCIPy coronagraph simulations. The root [README](../README.md) presents the main figures with metric definitions and interpretation.

| Output prefix | Contents |
| --- | --- |
| `validation_` | Shared-scale PSFs, radial profiles, charge-2 aberration comparisons, and sampling checks |
| `charge_mode_` | Charge 2/4/6 comparisons for defocus, astigmatism, and coma |
| `charge_offaxis_` | Relative core throughput versus source separation |
| `charge6_` | Charge-6 pupil-grid sampling sensitivity |
| `charge_stellar_` and `charge_ideal_` | Single-mode charge comparison and zero-aberration images |
| `charge_tradeoff_` | Illustrative throughput-to-leakage ratio, not an SNR |
| `contrast` and `science_images` | Initial single-mode demonstration |

Stellar leakage is normalized to the ideal unocculted reference PSF peak. Off-axis core throughput is relative to a matched Lyot-only reference. See the root README for the measurement apertures and limitations.

Run the scripts from the repository root to regenerate the outputs; this can overwrite the saved files. No simulations were rerun as part of the documentation update.
