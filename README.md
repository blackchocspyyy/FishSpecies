# Hydroacoustic-Based Fish Species Classification
### For Prof. Leos Barajas, please check the notebooks/ folder directly. It contains the most up-to-date code. The main python file is called "FishClassificationCode.ipynb", utilities are under notebooks/src/.

This project focuses on classifying Lake Trout and Smallmouth Bass using hydroacoustic data collected from fish provided by our collaborators. Multiple models were built and evaluated, including traditional machine learning models such as Logistic Regression, Random Forest, and XGBoost. While XGBoost achieved the highest performance among these methods (~72% accuracy), it was limited by its inability to capture the time-series structure in the data.

To better utilize the sequential nature of the hydroacoustic signals, a deep learning model based on Long Short-Term Memory (LSTM) networks was developed. The LSTM model achieved 73.8% accuracy under Leave-One-Pair-Out (LOPO) cross-validation, showing a clear improvement over traditional baselines. Although this result is slightly below the 80% target, it demonstrates that deep learning is effective at extracting meaningful temporal patterns from the data.

This project can be extended by incorporating advanced feature engineering, longer training, and experimenting with models such as Transformers to further improve accuracy.

The research extends previous work by Leivesley and Professor Leos Barajas (2024).

## Project Structure
hydroacoustic-fish-classification/ 
- data/ # Raw and processed datasets (.csv) 
- notebooks/ # Jupyter notebooks for model training and analysis 
    - archive/ # Archived exploratory or older versions 
    - src/ # Python modules for data, feature, and model utilities 
- images/ # Correlation heatmaps, model result visualizations, and presentation slides
- results/ # Output tables and summary metrics 
- report/ # LaTeX reports 
- .gitignore # Excluded files and folders 
- README.md # Project overview and structure (this file) 
- requirements.txt # Python package dependencies

## Reproducibility Note
**Accuracy Variance**: Model performance metrics may show slight variations (±1-2%) between runs due to random seeds.


**Run time**: Run time for full 63 LOPO pairs validation in LSTM will be more than 1 hour.
