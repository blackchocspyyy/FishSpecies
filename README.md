# Hydroacoustic-Based Fish Species Classification
### For Prof. Leos Barajas, please check the notebooks/ folder directly. It contains the most up-to-date code.

This project explores machine learning and deep learning approaches for fish species classification using hydroacoustic data. Traditional monitoring methods like trawling and netting are costly and disruptive to aquatic ecosystems. This study investigates Long Short-Term Memory (LSTM) models to classify Lake Trout (LT) and Smallmouth Bass (SMB) based on sound wave reflections at different frequencies.

The research extends previous work by Leivesley and Professor Leos Barajas (2024) and aims to improve classification accuracy to at least 80% while identifying key frequency patterns.

## Project Structure
hydroacoustic-fish-classification/ 
- data/ # Raw and processed datasets (.csv) 
- notebooks/ # Jupyter notebooks for model training and analysis 
    - archive/ # Archived exploratory or older versions 
    - src/ # Python modules for data, feature, and model utilities 
- figures/ # Correlation heatmaps, model result visualizations
- results/ # Output tables and summary metrics 
- report/ # Final LaTeX report and related assets 
- .gitignore # Excluded files and folders 
- README.md # Project overview and structure (this file) 
- requirements.txt # Python package dependencies

## Reproducibility Note
**Accuracy Variance**: Model performance metrics may show slight variations (±1-2%) between runs due to random seeds.


**Run time**: Run time for full 63 LOPO pairs validation in LSTM will be more than 1 hour.
