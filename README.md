# The Footprint of Intelligence
### AI-Assisted Land Cover Change Detection — Cambridge A14 Corridor, 2019–2025

![Classification Animation](outputs/classification_animated.gif)

> *"This is what the AI economy looks like from 786 kilometres above the Earth."*

This project deploys a machine learning pipeline to detect and quantify the physical 
expansion of AI infrastructure — data centres, logistics warehouses, and industrial 
development — across the Cambridge A14 corridor between 2019 and 2025, using 
multi-temporal Sentinel-1 and Sentinel-2 satellite imagery.

The project then turns the same accounting framework on itself, measuring the carbon, 
energy, and water cost of the classifier doing the detecting — and comparing it to the 
embodied carbon of the infrastructure it found.

---

## Table of Contents
- [The Research Question](#the-research-question)
- [Key Results](#key-results)
- [Study Area](#study-area)
- [Remote Sensing Methodology](#remote-sensing-methodology)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Interactive Map](#interactive-map)
- [Environmental Impact](#environmental-impact)
- [Limitations](#limitations)
- [Video Walkthrough](#video-walkthrough)
- [References](#references)

---

## The Research Question

Can a machine learning pipeline detect and quantify the conversion of agricultural 
land to AI-related infrastructure across the Cambridge A14 corridor between 2019 
and 2025, and what is the environmental cost of the detection pipeline compared to 
the infrastructure it detects?

---

## Key Results

| Model | Overall Accuracy | Cohen's Kappa | Weighted F1 |
|---|---|---|---|
| K-Means (unsupervised baseline) | 71.9% | 0.507 | 0.722 |
| Random Forest (supervised) | 75.8% | 0.597 | 0.786 |
| U-Net (deep learning) | 77.4% | 0.607 | 0.793 |

**Land cover change detected (2019–2025):**
- 35.32 km² of agricultural and greenfield land converted to built-up/urban
- Annual expansion rate: ~5.9 km²/year across the A14 corridor

**Environmental finding:**
- Pipeline carbon footprint: **0.106 kg CO₂eq**
- Embodied carbon of detected infrastructure: **~15,894,000 tonnes CO₂eq**
- The infrastructure being detected emits approximately **150 million times** more 
carbon than the pipeline used to detect it

---

## Study Area

The A14 corridor between Cambourne in the west and Waterbeach in the north-east 
of Cambridge — approximately 30×30 km. Key transformation sites include:

- **Bourn Airfield / Cambourne West** — former RAF airfield converting to logistics 
and business park from 2018 onwards
- **Bar Hill and Swavesey** — major Amazon, DHL and third-party logistics warehouses 
built 2020–2023, individual buildings exceeding 100,000 m²
- **Waterbeach New Town** — residential development on former Waterbeach Barracks 
from 2021

All data falls within ESA Copernicus tile **T30UYC** (UTM zone 30N).

---

## Remote Sensing Methodology

### Dual-Sensor Fusion

This project uses two satellite sensors, not one. The combination is deliberate:

**Sentinel-2 L2A** measures surface reflectance across 13 spectral bands at 
10–20m resolution. It sees colour and vegetation — but is blocked by cloud cover.

**Sentinel-1 SAR** uses radar, which penetrates cloud and measures surface roughness 
and dielectric properties. It sees texture and structure, and is particularly effective 
at separating metal warehouse rooftops from bare soil.

Together they separate classes that either sensor alone confuses.

### Spectral Indices

Five spectral indices were calculated from Sentinel-2:

**NDVI** (Normalised Difference Vegetation Index):

$$NDVI = \frac{NIR - Red}{NIR + Red}$$

**NDWI** (Normalised Difference Water Index, McFeeters 1996):

$$NDWI = \frac{Green - NIR}{Green + NIR}$$

**NDBI** (Normalised Difference Built-up Index):

$$NDBI = \frac{SWIR - NIR}{SWIR + NIR}$$

**BSI** (Bare Soil Index):

$$BSI = \frac{(SWIR + Red) - (NIR + Blue)}{(SWIR + Red) + (NIR + Blue)}$$

**EVI** (Enhanced Vegetation Index):

$$EVI = 2.5 \times \frac{NIR - Red}{NIR + 6 \times Red - 7.5 \times Blue + 1}$$

### Epochs

Summer median composites (June–August) for four epochs:
- 2019 — pre-development baseline
- 2021 — construction phase
- 2023 — completion and infill
- 2025 — present state

### Land Cover Classes

| Class ID | Class | Key spectral signature |
|---|---|---|
| 1 | Dense woodland | High NDVI, low NDBI |
| 2 | Cropland | Moderate NDVI, low NDBI |
| 3 | Grassland / pasture | Moderate NDVI, very low NDBI |
| 4 | Water | Positive NDWI |
| 5 | Wetland / floodplain | High NDWI, moderate NDVI |
| 6 | Built-up / urban | High NDBI, high SAR backscatter |

---

## Machine Learning Methodology

Three models of increasing complexity were compared:

### K-Means Clustering (Unsupervised Baseline)
K-Means partitions pixels into k clusters based on spectral similarity alone, 
requiring no labelled training data. This serves as the unsupervised baseline — 
demonstrating what structure exists in the data without any human annotation.

k=7 clusters were used, matched to our class scheme. Cluster-to-class assignment 
was performed by majority vote against WorldCover labels.

### Random Forest (Supervised Baseline)
Random Forest trains an ensemble of 200 decision trees on labelled pixels from 
ESA WorldCover 2020. It is interpretable — the feature importance plot reveals 
which spectral indices drive classification decisions.

Key hyperparameters: `n_estimators=200`, `max_depth=20`, `class_weight='balanced'`

### U-Net (Deep Learning)
U-Net is a convolutional neural network originally designed for medical image 
segmentation. Unlike Random Forest, which classifies each pixel independently, 
U-Net processes 64×64 pixel patches — learning spatial context from neighbouring 
pixels. This produces smoother, more coherent maps and significantly improves 
detection of the built-up class.

Architecture: ResNet18-inspired encoder-decoder with skip connections.
Input: 15-feature stack per pixel. Training: 20 epochs on Colab T4 GPU.

### Feature Stack

Each pixel is described by 15 features:

| Feature | Source | Description |
|---|---|---|
| NDVI, NDWI, NDBI, BSI, EVI | Sentinel-2 | Spectral indices |
| B2, B3, B4, B8, B11 | Sentinel-2 | Raw reflectance bands |
| VV, VH, VH/VV | Sentinel-1 SAR | Radar backscatter |
| GLCM contrast, homogeneity | Sentinel-2 NIR | Texture features |

### Spatial Validation

Training used the top 70% of the image (spatially); testing used the bottom 30%. 
This prevents spatial autocorrelation from inflating accuracy scores — a critical 
methodological step often overlooked in remote sensing ML studies.

---

## Repository Structure
```
footprint-of-intelligence/
├── README.md
├── ENVIRONMENTAL_IMPACT.md
├── environment.yml
├── notebooks/
│   ├── 00_data_download.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_labelling.ipynb
│   ├── 04_models.ipynb
│   ├── 05_change_detection.ipynb
│   └── 06_environmental_impact.ipynb
├── outputs/
│   ├── classification_animated.gif
│   ├── interactive_map.html
│   ├── change_map_2019_2025.png
│   ├── rf_confusion_matrix.png
│   ├── rf_feature_importance.png
│   └── area_statistics.png
└── data/
    └── README.md
```

---

## Getting Started

### Requirements
```bash
pip install earthengine-api rasterio scikit-learn torch torchvision 
pip install folium imageio codecarbon scikit-image
```

Or use the provided environment file:
```bash
conda env create -f environment.yml
conda activate footprint
```

### Data Access

Raw Sentinel-1 and Sentinel-2 data is not included due to file size (~5 GB). 
To reproduce the data download, run `00_data_download.ipynb` which queries 
Google Earth Engine directly. A GEE account is required (free at 
earthengine.google.com).

ESA WorldCover labels are downloaded automatically within `03_labelling.ipynb`.

### Running the Pipeline

Run notebooks in order: `00` → `01` → `02` → `03` → `04` → `05` → `06`

All notebooks are designed for Google Colab. File paths point to Google Drive 
at `GEOL0069/Project/Data/` — update these if running locally.

---

## Interactive Map

An interactive Folium map showing all four classification epochs as toggleable 
layers is available in `outputs/interactive_map.html`. Download and open in 
any web browser — no server required.

---

## Environmental Impact

Full analysis in [ENVIRONMENTAL_IMPACT.md](ENVIRONMENTAL_IMPACT.md)

| Cost component | Estimate |
|---|---|
| Total compute energy | 0.454 kWh |
| Pipeline carbon footprint | 105.7 gCO₂eq (0.106 kg) |
| Water consumed (cooling) | 0.50 litres |
| Data downloaded | ~5 GB |
| Equivalent driving distance | 0.5 km |

The pipeline's carbon footprint is compared against the embodied carbon of the 
35.32 km² of infrastructure detected — approximately 15.9 million tonnes CO₂eq. 
The albedo decline from cropland (0.22) to warehouse rooftop (0.10) over the 
converted area creates a local radiative forcing of +0.51 W/m², a permanent 
biophysical warming signal detectable only through satellite remote sensing.

---

## Limitations

- WorldCover 2020 labels used as ground truth for earlier epochs — label accuracy 
in peri-urban transition zones may be lower than in clearly defined classes
- A single seasonal composite per epoch introduces phenological variation 
(particularly visible in the 2025 image which captured more post-harvest fields)
- The U-Net was trained on 2025 labels and applied to earlier epochs — spectral 
differences between seasons may introduce some inconsistency in absolute area 
statistics. The transition matrix (2019 vs 2025 directly compared) is more 
reliable than per-epoch area totals
- Water and wetland classes are underrepresented in the study area, leading to 
lower F1 scores for these classes across all models

---

## Video Walkthrough

[Link to video — add your video link here]

---

## Contact

Ariella Morris - ariella.morris.23@ucl.ac.uk / ariellamorris2@gmail.com
Project Link: https://github.com/ariellamorris2-cmd/GEOL0069-ML-Cambridge-LULC-Change

---

## Acknowledgements

This project was created for GEOL0069 — AI for Earth Observation at the University 
College London. I would like to thank Dr Michel Tsamados and Weibin Chen for 
their teaching, guidance, and for designing a module that encouraged genuinely 
original thinking about the intersection of machine learning and environmental 
monitoring.

The dual environmental framing of this project — using AI to detect AI 
infrastructure, and then measuring the cost of doing so — emerged directly from 
the course's emphasis on critical engagement with the environmental impact of 
computational methods. I am grateful for that prompt.

ESA WorldCover data was used under an open licence. Sentinel-1 and Sentinel-2 
imagery was accessed via Google Earth Engine under the Copernicus Open Data Policy.

---

## References

ESA Sentinel-2 Mission. European Space Agency. https://sentinel.esa.int

ESA WorldCover 2020. https://esa-worldcover.org

McFeeters, S.K. (1996). The use of the Normalized Difference Water Index (NDWI) 
in the delineation of open water features. *International Journal of Remote Sensing*, 
17(7), 1425–1432.

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks 
for biomedical image segmentation. *MICCAI 2015*.

Liang, S. (2001). Narrowband to broadband conversions of land surface albedo. 
*Remote Sensing of Environment*, 76(2), 213–238.

RICS (2023). Whole Life Carbon Assessment for the Built Environment.

---

*GEOL0069 — AI for Earth Observation | University College London*
