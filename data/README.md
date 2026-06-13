## Data

The dataset is not committed to this repository due to size (~700MB compressed).

### Download

```bash
curl -L -o data/criteo-research-uplift-v2.1.csv.gz "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
Dataset Details
PropertyValueSourceCriteo AI LabFull URLhttps://ailab.criteo.com/criteo-uplift-prediction-datasetRows13,979,592Columns16 (f0–f11, treatment, visit, conversion, exposure)Size (compressed)~700MBSize (uncompressed)~2.1GBTreatment rate~84.6%Conversion rate~0.30%
Expected File Path
After downloading, place the file at:
causal-uplift/data/criteo-research-uplift-v2.1.csv.gz
The pipeline reads from this exact path. Do not rename the file.
Citation
Diemert, E., Betlei, A., Renaudin, C., & Amini, M.R. (2018).

A Large Scale Benchmark for Uplift Modeling.

ACM SIGKDD Workshop on Causal Discovery, Predictions and Decision Making.

