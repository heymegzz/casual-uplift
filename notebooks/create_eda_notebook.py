import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

cells = []

# Title & Introduction
cells.append(nbf.v4.new_markdown_cell("""# Causal Uplift Modeling: End-to-End Analysis
This notebook walks through the results of our heterogeneous treatment effect (HTE) modeling pipeline on the Criteo dataset.
We trained several meta-learners (S, T, X, R) and a Causal Forest to estimate individual-level treatment effects (CATE)."""))

# Setup
cells.append(nbf.v4.new_code_cell("""import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

# Load evaluation metrics
with open('../results/metrics.json', 'r') as f:
    metrics = json.load(f)"""))

# Display Metrics
cells.append(nbf.v4.new_markdown_cell("""## 1. Overall Model Performance
We evaluate the models using Area Under the Uplift Curve (AUUC) and the Qini coefficient."""))

cells.append(nbf.v4.new_code_cell("""# Format metrics into a clean DataFrame
records = []
for model_name, data in metrics.items():
    records.append({
        'Model': model_name,
        'AUUC': data['auuc'],
        'Qini': data['qini'],
        'Calibration Error': data['cal_error'],
        'Policy Value @ 5%': data['policy_values']['0.05']
    })

df_metrics = pd.DataFrame(records).sort_values('AUUC', ascending=False)
display(df_metrics.style.highlight_max(subset=['AUUC', 'Qini', 'Policy Value @ 5%'], color='lightgreen'))"""))

# Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 2. Propensity Score & Overlap
Ensuring sufficient overlap between the treated and control populations is a foundational assumption in causal inference (positivity)."""))
cells.append(nbf.v4.new_code_cell("display(Image(filename='../results/figures/01_propensity_overlap.png'))"))

cells.append(nbf.v4.new_markdown_cell("""## 3. Uplift & Qini Curves
These curves show how much cumulative incremental lift we get by targeting the top X% of users ranked by their predicted CATE."""))
cells.append(nbf.v4.new_code_cell("display(Image(filename='../results/figures/02_auuc_comparison.png'))"))
cells.append(nbf.v4.new_code_cell("display(Image(filename='../results/figures/03_qini_curves.png'))"))

cells.append(nbf.v4.new_markdown_cell("""## 4. Policy Values
Expected average treatment effect if we implement a policy targeting only the top fractions."""))
cells.append(nbf.v4.new_code_cell("display(Image(filename='../results/figures/06_policy_value_curves.png'))"))

cells.append(nbf.v4.new_markdown_cell("""## 5. Model Calibration
How well do the predicted CATEs align with the actual observed Average Treatment Effects (ATE) by decile?"""))
cells.append(nbf.v4.new_code_cell("display(Image(filename='../results/figures/04_cate_calibration.png'))"))

cells.append(nbf.v4.new_markdown_cell("""## 6. Feature Importance & Placebo Test
We run a placebo test by randomly shuffling the treatment assignments and retraining. The resulting AUUC drops near zero, confirming our real models are picking up genuine signal rather than noise."""))
cells.append(nbf.v4.new_code_cell("display(Image(filename='../results/figures/05_shap_summary.png'))"))
cells.append(nbf.v4.new_code_cell("display(Image(filename='../results/figures/07_placebo_test.png'))"))

nb['cells'] = cells

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully at notebooks/analysis.ipynb")
