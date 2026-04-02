# ExoMinerLite — Detección y Validación de Exoplanetas

Modelo de deep learning para detectar y validar candidatos a exoplanetas usando datos reales de Kepler DR25, inspirado en [ExoMiner](https://arxiv.org/abs/2111.10009) de NASA Ames.

## Resultados

| Métrica | Valor |
|---------|-------|
| **Test AUC-ROC** | **0.914** |
| Precisión (planeta) | 83% |
| Recall (planeta) | 83% |
| Arquitectura | ExoMinerLite (86K parámetros) |
| Datos de entrenamiento | 5,000 KOIs de Kepler |

### Candidatos Validados

4 Kepler Objects of Interest clasificados como probables planetas que pasaron todas las pruebas de validación:

| KOI | P(planeta) | Periodo | Radio | Validación |
|-----|-----------|---------|-------|------------|
| K01590.01 | 87.5% | 12.89 d | 1.9 R⊕ | 5/5 ✅ |
| K00353.03 | 83.7% | 11.16 d | 2.0 R⊕ | 5/5 ✅ |
| K02298.03 | 78.5% | 2.47 d | 0.4 R⊕ | 5/5 ✅ |
| K03426.01 | 70.5% | 58.77 d | 1.6 R⊕ | 4/5 ✅ |

## Arquitectura

CNN multi-rama con 6 flujos de entrada, inspirada en ExoMiner (NASA Ames, 2022):

```
Vista global  (2001 pts) ──→ CNN 3 bloques ──→ 64 features ─┐
Vista local   (201 pts)  ──→ CNN 3 bloques ──→ 64 features ─┤
Tránsitos odd (201 pts)  ──→ CNN 2 bloques ──→ 32 features ─┤
Tránsitos even(201 pts)  ──→ CNN 2 bloques ──→ 32 features ─┼──→ Fusión (240) ──→ 128 ──→ 64 ──→ 1
Secundario    (201 pts)  ──→ CNN 2 bloques ──→ 32 features ─┤
Params estelares (6)     ──→ FC 32→16      ──→ 16 features ─┘
```

### Vistas Diagnósticas

| Vista | Propósito |
|-------|-----------|
| **Global** | Periodo orbital completo — forma general del tránsito, profundidad, simetría |
| **Local** | Zoom en el tránsito — forma del ingress/egress, V vs U |
| **Odd/Even** | Detectar binarias eclipsantes (profundidades diferentes = binaria) |
| **Secundario** | Fase 0.5 — eclipse secundario = binaria, no planeta |
| **Estelar** | Teff, Rstar, logg, profundidad, SNR, parámetro de impacto |

## Pipeline de Validación

Cada candidato pasa por 5 pruebas:

1. **Cruce con catálogos** — Verificar si ya fue confirmado/descartado en el NASA Exoplanet Archive
2. **Análisis de centroide** — Verificar que la señal viene de la estrella target, no de una vecina
3. **Test odd/even** — Comparar profundidad de tránsitos pares vs impares (Mann-Whitney U)
4. **Eclipse secundario** — Buscar dip en fase 0.5
5. **Consistencia física** — Verificar que radio, profundidad y periodo sean físicamente posibles

## Pipeline Completo

```
NASA Exoplanet Archive (catálogo KOI)
        │
        ▼
MAST Archive (curvas de luz de Kepler vía lightkurve)
        │
        ▼
Preprocesamiento (5 vistas diagnósticas + median binning)
        │
        ▼
ExoMinerLite (clasificación CNN multi-rama)
        │
        ▼
Ranking de candidatos (P > 0.7 → validación)
        │
        ▼
Validación (centroide + odd/even + secundario + física)
        │
        ▼
Reporte (prioridad: HIGH / MEDIUM / LOW / REJECT)
```

## Instalación

### Requisitos

- Python 3.8+
- GPU NVIDIA (para entrenamiento; la inferencia corre en CPU)
- ~10 GB de espacio en disco (caché de lightkurve + datos procesados)

### Configuración

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install lightkurve astropy astroquery
pip install numpy pandas matplotlib scikit-learn scipy tqdm
```

### Ejecución

Abrir `astro_exominerlite_final.ipynb` en Jupyter y ejecutar todas las celdas.
La primera ejecución descarga ~5,000 curvas de luz de MAST (~4-5 horas).
Las ejecuciones posteriores cargan desde caché (~10 segundos).

## Fuentes de Datos

- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — Catálogo KOI con disposiciones
- [MAST Archive](https://archive.stsci.edu/) — Curvas de luz de Kepler vía lightkurve
- [Kepler DR25](https://exoplanetarchive.ipac.caltech.edu/docs/Q1Q17-DR25-KOIcompanion.html) — Última liberación de datos de Kepler

## Contexto de Rendimiento

| Modelo | AUC | Datos |
|--------|-----|-------|
| Este proyecto (sintético) | 0.997 | Solo datos sintéticos |
| Este proyecto (real, 1K) | 0.748 | Kepler DR25, 2 vistas |
| Este proyecto (real, 5K) | 0.783 | + augmentation |
| **Este proyecto (ExoMinerLite)** | **0.914** | **+ estelar + odd/even + secundario** |
| Astronet (Shallue 2018) | ~0.96 | Global + Local (34K TCEs) |
| ExoMiner (Valizadegan 2022) | ~0.99 | 9 vistas diagnósticas |

## Referencias

- Shallue & Vanderburg (2018) — [Identifying Exoplanets with Deep Learning](https://arxiv.org/abs/1712.05044)
- Ansdell et al. (2018) — [Scientific Domain Knowledge Improves Exoplanet Transit Classification](https://arxiv.org/abs/1810.14530)
- Valizadegan et al. (2022) — [ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier](https://arxiv.org/abs/2111.10009)
- Valizadegan et al. (2025) — [ExoMiner++ on TESS](https://arxiv.org/abs/2502.09790)

## Autor

**Mario Carvajal** — Costa Rica, 2026

## Licencia

MIT
