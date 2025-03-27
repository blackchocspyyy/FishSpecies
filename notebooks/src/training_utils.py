import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple, List
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import LSTM, Dense, Masking
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras import Input

def evaluate_lofo_models(
    df_filtered: pd.DataFrame,
    X_rf_selected: pd.DataFrame,
    X_pca: np.ndarray,
    models: Dict[str, object]
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    """
    Evaluate multiple models using Leave-One-Fish-Out (LOFO) cross-validation.

    Args:
        df_filtered (pd.DataFrame): DataFrame with species labels and fishNum groups.
        X_rf_selected (pd.DataFrame): Features selected by Random Forest.
        X_pca (np.ndarray): PCA-transformed features.
        models (Dict[str, object]): Dictionary of model_name: model_instance.

    Returns:
        Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
            - Raw accuracy results for each fold.
            - Summary DataFrame of mean accuracies.
    """
    logo = LeaveOneGroupOut()
    groups = df_filtered['fishNum']
    results = {model: {"accuracy": []} for model in models}

    for train_idx, test_idx in logo.split(df_filtered, df_filtered['species_label'], groups):
        X_train_rf, X_test_rf = X_rf_selected.iloc[train_idx], X_rf_selected.iloc[test_idx]
        X_train_pca, X_test_pca = X_pca[train_idx], X_pca[test_idx]
        y_train = df_filtered.loc[train_idx, 'species_label']
        y_test = df_filtered.loc[test_idx, 'species_label']

        scaler_rf = StandardScaler()
        X_train_rf_scaled = scaler_rf.fit_transform(X_train_rf)
        X_test_rf_scaled = scaler_rf.transform(X_test_rf)

        for model_name, model in models.items():
            if model_name == "RandomForest_Selected":
                X_train, X_test = X_train_rf_scaled, X_test_rf_scaled
            else:
                X_train, X_test = X_train_pca, X_test_pca

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[model_name]["accuracy"].append(accuracy_score(y_test, y_pred))

    summary = {
        model: {
            "Mean Accuracy": np.mean(results[model]["accuracy"])
        } for model in results
    }
    return results, pd.DataFrame(summary).T

def evaluate_lofo_xgboost_multi(
    df_filtered: pd.DataFrame,
    feature_sets: Dict[str, pd.DataFrame],
    groups: pd.Series
) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    """
    Evaluate multiple XGBoost models using different feature sets via LOFO cross-validation.

    Args:
        df_filtered (pd.DataFrame): DataFrame with species_label column.
        feature_sets (Dict[str, pd.DataFrame]): Dict of model name -> feature DataFrame.
        groups (pd.Series): Group identifiers (e.g., fishNum).

    Returns:
        Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
            - All accuracy results per model.
            - Summary DataFrame of mean accuracy.
    """
    results = {name: {"accuracy": []} for name in feature_sets}
    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(df_filtered, df_filtered['species_label'], groups):
        y_train = df_filtered.iloc[train_idx]['species_label']
        y_test = df_filtered.iloc[test_idx]['species_label']

        for name, X in feature_sets.items():
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42
            )

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, y_pred)
            results[name]["accuracy"].append(acc)

    summary = {
        name: {"Mean Accuracy": np.mean(results[name]["accuracy"])}
        for name in results
    }

    return results, pd.DataFrame(summary).T

def tune_xgboost_with_cv_multi(
    df_filtered: pd.DataFrame,
    feature_sets: Dict[str, pd.DataFrame],
    y: pd.Series,
    n_iter: int = 20
) -> pd.DataFrame:
    """
    Tune XGBoost hyperparameters for multiple feature sets using 5-fold CV.

    Args:
        df_filtered (pd.DataFrame): Main DataFrame (only used for indexing).
        feature_sets (Dict[str, pd.DataFrame]): Dict of model name → features.
        y (pd.Series): Target labels.
        n_iter (int): Number of search iterations.

    Returns:
        pd.DataFrame: Summary with best accuracy and parameters per model.
    """
    summary = []

    for name, X in feature_sets.items():
        print(f"Tuning XGBoost for: {name}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        param_dist = {
            "n_estimators": [100, 300, 500], 
            "max_depth": [3, 6, 10],  
            "learning_rate": [0.01, 0.1, 0.3],  
            "subsample": [0.6, 0.8, 1.0], 
            "colsample_bytree": [0.6, 0.8, 1.0],  
        }

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        search.fit(X_scaled, y)

        summary.append({
            "Model Name": name,
            "Best Accuracy": search.best_score_,
            "Best Parameters": search.best_params_
        })

    return pd.DataFrame(summary)

def evaluate_lofo_xgboost_smote(
    df_filtered: pd.DataFrame,
    top_features_rf: List[str]
) -> pd.DataFrame:
    """
    Evaluate XGBoost with SMOTE on statistical summaries of RF-selected features using LOFO CV.

    Args:
        df_filtered (pd.DataFrame): Full dataframe with raw features and labels.
        top_features_rf (List[str]): Feature names selected by Random Forest.

    Returns:
        pd.DataFrame: Summary with mean accuracy for SMOTE-enhanced model.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import accuracy_score
    import xgboost as xgb
    from scipy import stats

    grouped = df_filtered.groupby("fishNum")

    def extract_features(group):
        features = {}
        for col in top_features_rf:
            values = group[col].values
            features[f"{col}_mean"] = np.mean(values)
            features[f"{col}_std"] = np.std(values)
            features[f"{col}_skew"] = stats.skew(values)
            features[f"{col}_kurtosis"] = stats.kurtosis(values)
        return pd.Series(features)

    # Extract features
    df_features = grouped.apply(extract_features).reset_index()
    df_features = df_features.merge(
        df_filtered[['fishNum', 'species_label']].drop_duplicates(), on="fishNum"
    )

    X_new = df_features.drop(columns=["fishNum", "species_label"])
    y_new = df_features["species_label"]
    groups = df_features["fishNum"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new)

    logo = LeaveOneGroupOut()
    results = {"accuracy": []}

    for train_idx, test_idx in logo.split(X_scaled, y_new, groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_new.iloc[train_idx], y_new.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results["accuracy"].append(acc)

    return pd.DataFrame([{
        "Model Name": "XGB_RF_Stat+SMOTE",
        "Mean Accuracy": np.mean(results["accuracy"])
    }])


def evaluate_lofo_rf_smote(
    df_filtered: pd.DataFrame,
    top_features_rf: List[str]
) -> pd.DataFrame:
    """
    Evaluate Random Forest with SMOTE using statistical summaries of RF-selected features and LOFO.

    Args:
        df_filtered (pd.DataFrame): Original dataframe.
        top_features_rf (List[str]): Features selected via Random Forest.

    Returns:
        pd.DataFrame: Summary with LOFO accuracy.
    """

    grouped = df_filtered.groupby("fishNum")

    def extract_features(group):
        features = {}
        for col in top_features_rf:
            values = group[col].values
            features[f"{col}_mean"] = np.mean(values)
            features[f"{col}_std"] = np.std(values)
            features[f"{col}_skew"] = stats.skew(values)
            features[f"{col}_kurtosis"] = stats.kurtosis(values)
        return pd.Series(features)

    df_features = grouped.apply(extract_features).reset_index()
    df_features = df_features.merge(
        df_filtered[['fishNum', 'species_label']].drop_duplicates(), on="fishNum"
    )

    X_new = df_features.drop(columns=["fishNum", "species_label"])
    y_new = df_features["species_label"]
    groups = df_features["fishNum"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new)

    logo = LeaveOneGroupOut()
    results = {"accuracy": []}

    for train_idx, test_idx in logo.split(X_scaled, y_new, groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_new.iloc[train_idx], y_new.iloc[test_idx]

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=500, random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results["accuracy"].append(acc)

    return pd.DataFrame([{
        "Model Name": "RF_RF_Stat+SMOTE",
        "Mean LOFO Accuracy": np.mean(results["accuracy"])
    }])

def evaluate_lstm_lofo_kfold(
    fish_sequences_padded: np.ndarray,
    fish_labels_encoded: np.ndarray,
    fish_nums: List[str],
    lopo_pairs: List[Tuple[str, str]],
    random_lopo_pairs: List[Tuple[str, str]],
    input_shape: Tuple[int, int],
    use_all_pairs: bool = False
) -> pd.DataFrame:
    """
    Evaluate LSTM model using LOPO and Stratified K-Fold CV.

    Args:
        fish_sequences_padded (np.ndarray): 3D input data for LSTM.
        fish_labels_encoded (np.ndarray): Encoded labels.
        fish_nums (List[str]): Fish ID list.
        lopo_pairs (List[Tuple]): All (LT, SMB) test pairs.
        random_lopo_pairs (List[Tuple]): Random subset for fast testing.
        input_shape (Tuple): Shape of LSTM input (timesteps, features).
        use_all_pairs (bool): If True, run on all LOPO pairs.

    Returns:
        pd.DataFrame: Summary table with LOPO and K-Fold accuracies.
    """


    def build_lstm_model(input_shape):
        model = Sequential([
            Input(shape=input_shape),              
            Masking(mask_value=0.0),              
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model


    # --- LOPO ---
    results_lopo = []
    pairs = random_lopo_pairs if not use_all_pairs else lopo_pairs
    print(f"Running LOPO on {len(pairs)} pairs...")


    for lt_fish, smb_fish in pairs:
        test_indices = [fish_nums.index(lt_fish), fish_nums.index(smb_fish)]
        train_indices = [i for i in range(len(fish_nums)) if i not in test_indices]

        X_train = fish_sequences_padded[train_indices]
        X_test = fish_sequences_padded[test_indices]
        y_train = fish_labels_encoded[train_indices]
        y_test = fish_labels_encoded[test_indices]

        model = build_lstm_model(input_shape)
        model.fit(X_train, y_train, epochs=10, batch_size=2, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
        acc = accuracy_score(y_test, y_pred)
        results_lopo.append(acc)

    # --- K-Fold ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results_kfold = []

    for train_idx, test_idx in skf.split(fish_sequences_padded, fish_labels_encoded):
        X_train = fish_sequences_padded[train_idx]
        X_test = fish_sequences_padded[test_idx]
        y_train = fish_labels_encoded[train_idx]
        y_test = fish_labels_encoded[test_idx]

        model = build_lstm_model(input_shape)
        model.fit(X_train, y_train, epochs=10, batch_size=2, verbose=0)
        y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()
        acc = accuracy_score(y_test, y_pred)
        results_kfold.append(acc)

    return pd.DataFrame([{
        "Model Name": "LSTM",
        "Mean LOPO Accuracy": np.mean(results_lopo),
        "Mean K-Fold Accuracy": np.mean(results_kfold)
    }])
