# ExoMinerLite — Exoplanet Detection & Validation

Deep learning model for detecting and validating exoplanet candidates using real Kepler DR25 data, inspired by NASA's [ExoMiner](https://arxiv.org/abs/2111.10009).

## Results

| Metric | Value |
|--------|-------|
| **Test AUC-ROC** | **0.914** |
| Precision (planet) | 83% |
| Recall (planet) | 83% |
| Architecture | ExoMinerLite (86K params) |
| Training data | 5,000 Kepler KOIs |

### Validated Candidates

4 Kepler Objects of Interest classified as probable planets and passing all validation tests:

| KOI | P(planet) | Period | Radius | Validation |
|-----|-----------|--------|--------|------------|
| K01590.01 | 87.5% | 12.89 d | 1.9 R⊕ | 5/5 ✅ |
| K00353.03 | 83.7% | 11.16 d | 2.0 R⊕ | 5/5 ✅ |
| K02298.03 | 78.5% | 2.47 d | 0.4 R⊕ | 5/5 ✅ |
| K03426.01 | 70.5% | 58.77 d | 1.6 R⊕ | 4/5 ✅ |

## Architecture

Multi-branch CNN with 6 input streams, inspired by ExoMiner (NASA Ames, 2022):

```
Global view  (2001 pts) ──→ CNN 3 blocks ──→ 64 features ─┐
Local view   (201 pts)  ──→ CNN 3 blocks ──→ 64 features ─┤
Odd transits (201 pts)  ──→ CNN 2 blocks ──→ 32 features ─┤
Even transits(201 pts)  ──→ CNN 2 blocks ──→ 32 features ─┼──→ Fusion (240) ──→ 128 ──→ 64 ──→ 1
Secondary    (201 pts)  ──→ CNN 2 blocks ──→ 32 features ─┤
Stellar params (6)      ──→ FC 32→16     ──→ 16 features ─┘
```

### Diagnostic Views

| View | Purpose |
|------|---------|
| **Global** | Full orbital period — overall transit shape, depth, symmetry |
| **Local** | Zoom on transit — ingress/egress shape, V vs U |
| **Odd/Even** | Detect eclipsing binaries (different depths = binary) |
| **Secondary** | Phase 0.5 — secondary eclipse = binary, not planet |
| **Stellar** | Teff, Rstar, logg, depth, SNR, impact parameter |

## Validation Pipeline

Each candidate passes 5 tests:

1. **Archive cross-match** — Check if already confirmed/rejected in NASA Exoplanet Archive
2. **Centroid analysis** — Verify signal comes from target star, not a neighbor
3. **Odd/even test** — Compare depth of odd vs even transits (Mann-Whitney U)
4. **Secondary eclipse** — Search for dip at phase 0.5
5. **Physical consistency** — Verify radius, depth, period are physically possible

## Pipeline

```
NASA Exoplanet Archive (KOI catalog)
        │
        ▼
MAST Archive (Kepler light curves via lightkurve)
        │
        ▼
Preprocessing (5 diagnostic views + median binning)
        │
        ▼
ExoMinerLite (CNN multi-branch classification)
        │
        ▼
Candidate ranking (P > 0.7 → validation)
        │
        ▼
Validation (centroid + odd/even + secondary + physics)
        │
        ▼
Report (priority: HIGH / MEDIUM / LOW / REJECT)
```

## Setup

### Requirements

- Python 3.8+
- NVIDIA GPU (for training; inference runs on CPU)
- ~10 GB disk space (lightkurve cache + processed data)

### Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install lightkurve astropy astroquery
pip install numpy pandas matplotlib scikit-learn scipy tqdm
```

### Running

Open `astro_exominerlite_final.ipynb` in Jupyter and run all cells.
The first run downloads ~5,000 light curves from MAST (~4-5 hours).
Subsequent runs load from cache (~10 seconds).

## Data Sources

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — KOI catalog with dispositions
- [MAST Archive](https://archive.stsci.edu/) — Kepler light curves via lightkurve
- [Kepler DR25](https://exoplanetarchive.ipac.caltech.edu/docs/Q1Q17-DR25-KOIcompanion.html) — Final Kepler data release

## Performance Context

| Model | AUC | Data |
|-------|-----|------|
| This project (synthetic) | 0.997 | Synthetic only |
| This project (real, 1K) | 0.748 | Kepler DR25, 2 views |
| This project (real, 5K) | 0.783 | + augmentation |
| **This project (ExoMinerLite)** | **0.914** | **+ stellar + odd/even + secondary** |
| Astronet (Shallue 2018) | ~0.96 | Global + Local (34K TCEs) |
| ExoMiner (Valizadegan 2022) | ~0.99 | 9 diagnostic views |

## References

- Shallue & Vanderburg (2018) — [Identifying Exoplanets with Deep Learning](https://arxiv.org/abs/1712.05044)
- Ansdell et al. (2018) — [Scientific Domain Knowledge Improves Exoplanet Transit Classification](https://arxiv.org/abs/1810.14530)
- Valizadegan et al. (2022) — [ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier](https://arxiv.org/abs/2111.10009)
- Valizadegan et al. (2025) — [ExoMiner++ on TESS](https://arxiv.org/abs/2502.09790)

## Author

**Mario Carvajal** — Costa Rica, 2026

## License

MIT
