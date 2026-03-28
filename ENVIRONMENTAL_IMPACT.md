# Environmental Impact Assessment

## The Central Argument

This project uses an AI pipeline — trained on cloud compute, powered by data centres 
— to detect the physical expansion of data centres and warehouses across the Cambridge 
fringe. The classifier is both the instrument of analysis and a participant in the 
system it is analysing. That irony is not a footnote; it is the thesis.

---

## Part A — Pipeline Footprint

### Compute Energy

| Stage | Runtime | Energy (Wh) | Carbon (gCO₂eq) |
|---|---|---|---|
| Data download — Sentinel-2 | 15 min | 92.1 | 21.5 |
| Data download — Sentinel-1 | 5 min | 36.7 | 8.5 |
| Preprocessing | 20 min | 5.5 | 1.3 |
| Feature engineering | 25 min | 6.9 | 1.6 |
| Labelling (WorldCover) | 5 min | 13.5 | 3.1 |
| K-Means training | 3 min | 0.8 | 0.2 |
| Random Forest training | 4 min | 1.1 | 0.3 |
| U-Net training (T4 GPU) | 25 min | 32.1 | 7.5 |
| Inference — all epochs | 20 min | 25.7 | 6.0 |
| Change detection | 10 min | 2.8 | 0.6 |
| **TOTAL** | **132 min** | **216.2 Wh** | **50.4 gCO₂eq** |

Carbon intensity: UK grid average 233 gCO₂eq/kWh (DESNZ 2024).
Google data centre PUE: 1.10.

CodeCarbon was used to track emissions during model training cells.

### Hidden Costs

**Storage footprint:**
- Raw data: ~6 GB
- Processed data and feature stacks: ~3 GB
- Outputs: ~1 GB
- Total: ~10 GB stored on Google Drive for approximately 1 month

**Network transfer:**
- ~6 GB downloaded from Copernicus/GEE servers
- Network energy: 0.06 kWh/GB × 6 GB = 0.36 kWh = 360 Wh
- Network carbon: ~83.9 gCO₂eq

**Water footprint:**
- Google published Water Usage Effectiveness (WUE): ~1.1 litres/kWh
- Total compute: ~0.454 kWh
- Water consumed for cooling: **~0.50 litres**
- Context: a single large data centre consumes ~150,000 litres per day

### Total Pipeline Summary

| Metric | Value |
|---|---|
| Total energy | 0.454 kWh |
| Total carbon | 105.7 gCO₂eq (0.106 kgCO₂eq) |
| Water consumed | ~0.50 litres |
| Data downloaded | ~6 GB |
| Equivalent driving distance | ~0.5 km in a petrol car |

---

## Part B — Infrastructure Footprint

The built-up expansion detected by the classifier (35.32 km²) represents real 
physical construction. Every square metre has an associated embodied carbon cost.

**Source:** RICS Whole Life Carbon Assessment 2023
**Figure:** 380–520 kgCO₂eq per m² for UK logistics warehouse construction

| Estimate | Carbon |
|---|---|
| Low (380 kgCO₂eq/m²) | 13,421,600 tonnes CO₂eq |
| Mid (450 kgCO₂eq/m²) | 15,894,000 tonnes CO₂eq |
| High (520 kgCO₂eq/m²) | 18,366,400 tonnes CO₂eq |

**Important caveat:** This estimate assumes all converted land represents new 
warehouse construction and should be treated as an upper bound. Not all 35.32 km² 
represents building footprint — roads, car parks, and landscaping are included. 
Even assuming only 10% constitutes actual building construction, the embodied 
carbon remains approximately 1.5 million tonnes CO₂eq.

---

## Part C — The Central Comparison

| | Value |
|---|---|
| Pipeline carbon | 0.106 kgCO₂eq |
| Infrastructure embodied carbon (mid estimate) | 15,894,000,000 kgCO₂eq |
| **Ratio** | **~150,000,000,000 : 1** |

The infrastructure being detected emits approximately **150 billion times** more 
carbon than the pipeline used to detect it. Remote sensing is not carbon neutral, 
but it is extraordinarily carbon-efficient compared to the physical reality it monitors.

---

## Part D — Albedo Radiative Forcing

Converting agricultural land to warehouse rooftops changes the surface albedo — 
the fraction of solar radiation reflected back to space.

| Surface | Albedo |
|---|---|
| UK arable cropland | ~0.22 |
| Dark metal warehouse rooftop | ~0.10 |
| **Decline** | **0.12** |

**Calculation:**
- Converted area: 35.32 km² = 35,320,000 m²
- Mean annual incoming solar radiation (Cambridge): ~120 W/m²
- Additional energy absorbed: 35,320,000 × 120 × 0.12 = **507,408,000 W = 0.51 GW**
- Local radiative forcing over study area: **+0.514 W/m²**

This warming signal persists regardless of the operational carbon emissions of 
the warehouses themselves. It is a permanent biophysical change to the local 
energy budget, detectable only through satellite remote sensing.

For context, the total global radiative forcing from all human greenhouse gas 
emissions is approximately 3.3 W/m². The land conversion in this study area 
alone creates a local forcing of over 0.5 W/m².

---

## Part E — Green AI Practices

The following practices were adopted during this project to minimise environmental impact:

- Colab sessions disconnected immediately after each notebook completed
- Small spatial subsets used during code development and debugging
- Full-resolution processing only run once code was verified
- Data downloaded once and cached to Google Drive rather than re-downloading
- U-Net trained for 20 epochs only — loss curve confirmed convergence before this point

---

## References

Aslan, J. et al. (2018). Electricity Intensity of Internet Data Transmission. 
*Journal of Industrial Ecology*, 22(4), 785–798.

DESNZ (2024). UK grid carbon intensity. Department for Energy Security and Net Zero.

Google (2023). Environmental Report. Google LLC.

RICS (2023). Whole Life Carbon Assessment for the Built Environment, 2nd edition.

Liang, S. (2001). Narrowband to broadband conversions of land surface albedo. 
*Remote Sensing of Environment*, 76(2), 213–238.
