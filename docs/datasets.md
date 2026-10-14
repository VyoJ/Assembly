# Example datasets

Candidate datasets for the training demo (vision + telemetry/vibration).
The data itself is NOT versioned in git (see `.gitignore`); it is only documented here.

| # | Dataset | Link | Notes |
|---|---------|------|-------|
| 1 | MAFAULDA (Machinery Fault Database, UFRJ) | https://www02.smt.ufrj.br/~offshore/mfs/page_01.html | Vibration/audio recorded on a machinery fault simulator (imbalance, misalignment, bearing faults). 1951 time series, 50 kHz, 13 GB full set. |
| 2 | Mendeley Data `xjmtb22pff` | https://data.mendeley.com/datasets/xjmtb22pff/1 | Video dataset of safe/unsafe worker behaviours in a factory (691 1080p clips, CC BY 4.0). Workplace safety, not machine faults. Manual download. |
| 3 | MVTec AD | https://www.mvtec.com/research-teaching/datasets/mvtec-ad | Industrial visual anomaly detection (defects in objects/textures). Vision. CC BY-NC-SA; download requires a form. |
| 4 | Zenodo 3384388 (MIMII) | https://zenodo.org/records/3384388 | Sound dataset of valves, pumps, fans and slide rails, normal and anomalous. Audio. CC BY-SA 4.0; 7-11 GB per zip, ~100 GB total. |
| 5 | CWRU Bearing Data Center | https://engineering.case.edu/bearingdatace | Bearing vibration with seeded faults (`.mat` files). Time series / diagnosis. |

## Tentative mapping to tasks

- **Vision (defect detection):** MVTec AD
- **Time series / predictive maintenance:** MAFAULDA, CWRU
- **Audio:** MIMII (Zenodo)
- **Video / safety:** Mendeley dataset

The original project brief also mentions NASA C-MAPSS and SWaT as alternatives for RUL/telemetry.
