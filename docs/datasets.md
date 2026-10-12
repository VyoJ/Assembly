# Datasets de ejemplo

Datasets candidatos para el demo de training (visión + telemetría/vibración).
Los datos en sí NO se versionan en git (ver `.gitignore`); solo se documentan aquí.

| # | Dataset | Enlace | Notas |
|---|---------|--------|-------|
| 1 | MAFAULDA (Machinery Fault Database, UFRJ) | https://www02.smt.ufrj.br/~offshore/mfs/page_01.html | Vibración/sensores de banco de pruebas de maquinaria con fallos (desbalanceo, desalineación...). Series temporales. |
| 2 | Mendeley Data `xjmtb22pff` | https://data.mendeley.com/datasets/xjmtb22pff/1 | Por revisar (contenido sin verificar). |
| 3 | MVTec AD | https://www.mvtec.com/research-teaching/datasets/mvtec-ad | Detección de anomalías visuales industriales (defectos en objetos/texturas). Visión. |
| 4 | Zenodo 3384388 | https://zenodo.org/records/3384388 | Por revisar (contenido sin verificar). |
| 5 | CWRU Bearing Data Center | https://engineering.case.edu/bearingdatace | Vibración de rodamientos con fallos inyectados. Series temporales / diagnóstico. |

## Mapeo tentativo a tareas

- **Visión (defect detection):** MVTec AD
- **Series temporales / mantenimiento predictivo:** MAFAULDA, CWRU, (+ los dos por revisar)

El PDF original también menciona NASA C-MAPSS y SWaT como alternativas para RUL/telemetría.
