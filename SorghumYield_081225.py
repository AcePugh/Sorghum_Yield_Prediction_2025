import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
import platform
import pickle
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
from tkinter.simpledialog import askinteger, askstring
from tkinter import messagebox
from datetime import datetime
from fpdf import FPDF
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

root = tk.Tk()
root.withdraw()

use_pickle = messagebox.askyesno("Mode Selection", "Do you want to use saved model(s) from pickle file(s) for prediction?")

if use_pickle:
    pickle_path = askopenfilename(title="Select the model pickle file",
                                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
    if not pickle_path:
        raise SystemExit("No model file selected.")
    
    print(f"[INFO] Loading model from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        model = pickle.load(f)
    
    model_name = os.path.basename(pickle_path).replace('_best_model.pkl', '').replace('_', ' ').title()
    print(f"[INFO] Loaded {model_name} model")
    
    file_path = askopenfilename(title="Select the CSV file with data for prediction")
    if not file_path:
        raise SystemExit("No data file selected.")
    print(f"[INFO] Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    encoded_cols_str = askstring("One-Hot Encoding",
                            "Enter columns to one-hot encode (comma-separated) or leave blank:")
    columns_to_encode = []
    if encoded_cols_str and encoded_cols_str.strip():
        columns_to_encode = [col.strip() for col in encoded_cols_str.split(",") if col.strip()]
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)
    print("[INFO] One-hot encoding complete." if columns_to_encode else "[INFO] No one-hot columns specified.")
    
    base_results_dir = askdirectory(title="Select output folder for predictions")
    if not base_results_dir:
        raise SystemExit("No folder selected.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = os.path.join(base_results_dir, f"predictions_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n[INFO] Results will be saved to: {results_dir}")
    
    standard_drop_cols = ['RGB_Path', 'DSM_Path', 'img_filelist']
    
    has_target = messagebox.askyesno("Target Column",
                                    "Does your data include the target column (weight_lb)?")
    
    drop_cols_str = askstring("Additional Columns to Drop",
                            "Enter additional columns to drop (comma-separated) or leave blank:")
    drop_cols = standard_drop_cols.copy()
    if drop_cols_str and drop_cols_str.strip():
        additional_drop_cols = [col.strip() for col in drop_cols_str.split(",") if col.strip()]
        drop_cols.extend(additional_drop_cols)
    
    print(f"[INFO] Will drop these columns: {drop_cols}")
        
    filename_col = None
    if 'filename' in df.columns:
        filename_col = df['filename'].copy()
        df = df.drop(columns=['filename'])
    
    target = 'weight_lb'
    if has_target:
        X = df.drop(columns=[target] + drop_cols, errors='ignore')
        y = df[target]
    else:
        X = df.drop(columns=drop_cols, errors='ignore')
        y = None
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_numeric = X[numeric_cols]
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    
    X_df_numeric = pd.DataFrame(X_numeric_scaled, columns=numeric_cols)
    X_non_numeric = X.drop(columns=numeric_cols).reset_index(drop=True)
    X_df = pd.concat([X_df_numeric, X_non_numeric], axis=1)
    
    predictions = model.predict(X_df[numeric_cols])
    
    predictions = np.maximum(predictions, 0)
    print("[INFO] Ensuring all weight predictions are non-negative")
    
    results = []
    if filename_col is not None:
        for i, (fname, pred) in enumerate(zip(filename_col, predictions)):
            row = {"Filename": fname, "Predicted_weight_lb": pred}
            if has_target:
                row["Actual_weight_lb"] = y.iloc[i]
            results.append(row)
    else:
        for i, pred in enumerate(predictions):
            row = {"ID": i+1, "Predicted_weight_lb": pred}
            if has_target:
                row["Actual_weight_lb"] = y.iloc[i]
            results.append(row)
    
    predictions_df = pd.DataFrame(results)
    predictions_path = os.path.join(results_dir, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"[INFO] Predictions saved to: {predictions_path}")
    
    if has_target:
        plt.figure(figsize=(10, 8))
        plt.scatter(y, predictions, alpha=0.7)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
        plt.xlabel('Actual Weight (lb)')
        plt.ylabel('Predicted Weight (lb)')
        plt.title(f'Actual vs. Predicted using {model_name}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "actual_vs_predicted.png"))
        plt.clf()
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        summary_path = os.path.join(results_dir, "prediction_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"Prediction Summary using {model_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Model file: {os.path.basename(pickle_path)}\n")
            f.write(f"Data file: {os.path.basename(file_path)}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"R² Score: {r2:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MSE: {mse:.4f}\n")
        
        print(f"[INFO] Prediction summary saved to: {summary_path}")
    
    print("[INFO] Prediction complete! 🎉")
    print(f"Results saved to: {results_dir}")
else:
    file_path = askopenfilename(title="Select the CSV file")
    if not file_path:
        raise SystemExit("No file selected.")
    print(f"[INFO] Loading data from: {file_path}")
    df = pd.read_csv(file_path)

    encoded_cols_str = askstring("One-Hot Encoding", "Enter columns to one-hot encode (comma-separated) or leave blank:")
    columns_to_encode = []
    if encoded_cols_str and encoded_cols_str.strip():
        columns_to_encode = [col.strip() for col in encoded_cols_str.split(",") if col.strip()]
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=False)
    print("[INFO] One-hot encoding complete." if columns_to_encode else "[INFO] No one-hot columns specified.")

    base_results_dir = askdirectory(title="Select output folder")
    if not base_results_dir:
        raise SystemExit("No folder selected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = os.path.join(base_results_dir, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n[INFO] Results will be saved to: {results_dir}")

    env_info = {
        "Python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "shap": shap.__version__,
        "xgboost": xgb.__version__,
        "catboost": "Not Used"
    }
    with open(os.path.join(results_dir, "environment.json"), "w") as f:
        json.dump(env_info, f, indent=2)

    print("[INFO] Loading data...")

    filename_col = None
    if 'filename' in df.columns:
        filename_col = df['filename'].copy()
        df = df.drop(columns=['filename'])

    target = 'weight_lb'
    drop_cols = ['RGB_Path', 'DSM_Path', 'img_filelist']
    X = df.drop(columns=[target] + drop_cols, errors='ignore')
    y = df[target]

    if filename_col is not None:
        X_train, X_test, y_train, y_test, fname_train, fname_test = train_test_split(
            X,
            y,
            filename_col,
            test_size=0.2,
            random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
        fname_train = fname_test = None

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train_numeric = X_train[numeric_cols]
    X_test_numeric = X_test[numeric_cols]

    scaler = StandardScaler()
    X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
    X_test_numeric_scaled = scaler.transform(X_test_numeric)

    X_train_df_numeric = pd.DataFrame(X_train_numeric_scaled, columns=numeric_cols)
    X_test_df_numeric = pd.DataFrame(X_test_numeric_scaled, columns=numeric_cols)

    X_train_non_numeric = X_train.drop(columns=numeric_cols).reset_index(drop=True)
    X_test_non_numeric = X_test.drop(columns=numeric_cols).reset_index(drop=True)
    X_train_df = pd.concat([X_train_df_numeric, X_train_non_numeric], axis=1)
    X_test_df = pd.concat([X_test_df_numeric, X_test_non_numeric], axis=1)

    range_start = askinteger("Random States (Start)", "Enter a start value for random states:")
    range_end = askinteger("Random States (End)", "Enter an end value for random states:")
    if range_start is None or range_end is None:
        raise SystemExit("No valid range provided.")
    random_states = range(range_start, range_end + 1)

    def train_and_select_best_model(base_model, param_grid, model_name):
        best_model = None
        best_test_r2 = -9999
        best_test_rmse = float("inf")
        best_rs = None

        r2_list = []
        rmse_list = []

        for rs in random_states:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=rs)

            X_tr_num = X_tr[numeric_cols]
            X_te_num = X_te[numeric_cols]
            scaler_loop = StandardScaler()
            X_tr_num_scaled = scaler_loop.fit_transform(X_tr_num)
            X_te_num_scaled = scaler_loop.transform(X_te_num)

            X_tr_df = pd.concat([
                pd.DataFrame(X_tr_num_scaled, columns=numeric_cols).reset_index(drop=True),
                X_tr.drop(columns=numeric_cols).reset_index(drop=True)
            ], axis=1)
            X_te_df = pd.concat([
                pd.DataFrame(X_te_num_scaled, columns=numeric_cols).reset_index(drop=True),
                X_te.drop(columns=numeric_cols).reset_index(drop=True)
            ], axis=1)

            params = base_model.get_params()
            if 'random_state' in params:
                base_model.set_params(random_state=rs)

            grid = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='neg_root_mean_squared_error',
                return_train_score=True
            )
            grid.fit(X_tr_df[numeric_cols], y_tr)
            candidate = grid.best_estimator_

            test_preds = candidate.predict(X_te_df[numeric_cols])
            test_r2 = r2_score(y_te, test_preds)
            test_rmse = np.sqrt(mean_squared_error(y_te, test_preds))

            r2_list.append(test_r2)
            rmse_list.append(test_rmse)

            if test_r2 > best_test_r2:
                best_test_r2 = test_r2
                best_test_rmse = test_rmse
                best_model = candidate
                best_rs = rs

        mean_r2 = np.mean(r2_list)
        std_r2 = np.std(r2_list, ddof=1)
        mean_rmse = np.mean(rmse_list)
        std_rmse = np.std(rmse_list, ddof=1)

        return best_model, best_rs, best_test_r2, best_test_rmse, mean_r2, std_r2, mean_rmse, std_rmse

    print("[INFO] Starting model training with multiple random states...")
    results = []
    best_models = {}

    print("[INFO] Training Ridge Regression...")
    ridge_best, ridge_rs, ridge_test_r2, ridge_test_rmse, ridge_mean_r2, ridge_std_r2, ridge_mean_rmse, ridge_std_rmse = train_and_select_best_model(
        Ridge(),
        {'alpha': [0.1, 1.0, 10.0]},
        "Ridge"
    )
    results.append({
        "Model": "Ridge",
        "Best Random State": ridge_rs,
        "Best Test R2": ridge_test_r2,
        "Best Test RMSE": ridge_test_rmse,
        "Mean R2": ridge_mean_r2,
        "Std R2": ridge_std_r2,
        "Mean RMSE": ridge_mean_rmse,
        "Std RMSE": ridge_std_rmse
    })
    best_models["Ridge"] = ridge_best

    print("[INFO] Training Lasso Regression...")
    lasso_best, lasso_rs, lasso_test_r2, lasso_test_rmse, lasso_mean_r2, lasso_std_r2, lasso_mean_rmse, lasso_std_rmse = train_and_select_best_model(
        Lasso(max_iter=1000000, tol=1e-4),
        {'alpha': [0.001, 0.01, 0.1, 1.0]},
        "Lasso"
    )
    results.append({
        "Model": "Lasso",
        "Best Random State": lasso_rs,
        "Best Test R2": lasso_test_r2,
        "Best Test RMSE": lasso_test_rmse,
        "Mean R2": lasso_mean_r2,
        "Std R2": lasso_std_r2,
        "Mean RMSE": lasso_mean_rmse,
        "Std RMSE": lasso_std_rmse
    })
    best_models["Lasso"] = lasso_best

    print("[INFO] Training ElasticNet...")
    enet_best, enet_rs, enet_test_r2, enet_test_rmse, enet_mean_r2, enet_std_r2, enet_mean_rmse, enet_std_rmse = train_and_select_best_model(
        ElasticNet(max_iter=1000000, tol=1e-4),
        {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        "ElasticNet"
    )
    results.append({
        "Model": "ElasticNet",
        "Best Random State": enet_rs,
        "Best Test R2": enet_test_r2,
        "Best Test RMSE": enet_test_rmse,
        "Mean R2": enet_mean_r2,
        "Std R2": enet_std_r2,
        "Mean RMSE": enet_mean_rmse,
        "Std RMSE": enet_std_rmse
    })
    best_models["ElasticNet"] = enet_best

    print("[INFO] Training Random Forest...")
    rf_best, rf_rs, rf_test_r2, rf_test_rmse, rf_mean_r2, rf_std_r2, rf_mean_rmse, rf_std_rmse = train_and_select_best_model(
        RandomForestRegressor(n_jobs=-1),
        {'n_estimators': [100, 200, 300, 400], 'max_depth': [5, 10, 20, 30, None]},
        "Random Forest"
    )
    results.append({
        "Model": "Random Forest",
        "Best Random State": rf_rs,
        "Best Test R2": rf_test_r2,
        "Best Test RMSE": rf_test_rmse,
        "Mean R2": rf_mean_r2,
        "Std R2": rf_std_r2,
        "Mean RMSE": rf_mean_rmse,
        "Std RMSE": rf_std_rmse
    })
    best_models["Random Forest"] = rf_best

    print("[INFO] Training XGBoost (GPU)...")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Falling back to prediction using DMatrix due to mismatched devices"
        )
        xgb_best, xgb_rs, xgb_test_r2, xgb_test_rmse, xgb_mean_r2, xgb_std_r2, xgb_mean_rmse, xgb_std_rmse = train_and_select_best_model(
            xgb.XGBRegressor(
                tree_method='gpu_hist',
                predictor='gpu_predictor'
            ),
            {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            "XGBoost"
        )
    results.append({
        "Model": "XGBoost",
        "Best Random State": xgb_rs,
        "Best Test R2": xgb_test_r2,
        "Best Test RMSE": xgb_test_rmse,
        "Mean R2": xgb_mean_r2,
        "Std R2": xgb_std_r2,
        "Mean RMSE": xgb_mean_rmse,
        "Std RMSE": xgb_std_rmse
    })
    best_models["XGBoost"] = xgb_best

    print("[INFO] Training Support Vector Machine...")
    svm_best, svm_rs, svm_test_r2, svm_test_rmse, svm_mean_r2, svm_std_r2, svm_mean_rmse, svm_std_rmse = train_and_select_best_model(
        SVR(),
        {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        },
        "SVR"
    )
    results.append({
        "Model": "SVR",
        "Best Random State": svm_rs,
        "Best Test R2": svm_test_r2,
        "Best Test RMSE": svm_test_rmse,
        "Mean R2": svm_mean_r2,
        "Std R2": svm_std_r2,
        "Mean RMSE": svm_mean_rmse,
        "Std RMSE": svm_std_rmse
    })
    best_models["SVR"] = svm_best

    results_df = pd.DataFrame(results).sort_values(by="Best Test R2", ascending=False)
    results_df.to_csv(os.path.join(results_dir, "model_selection_summary.csv"), index=False)
    print("\n[INFO] Model selection summary (by Best Test R2):")
    print(results_df)

    top_model_name = results_df.iloc[0]['Model']
    top_model = best_models[top_model_name]

    print(f"\n[INFO] Best overall model: {top_model_name}")

    print("[INFO] Saving best models as pickle files...")
    for model_name, model in best_models.items():
        model_filename = model_name.lower().replace(' ', '_') + '_best_model.pkl'
        pickle_path = os.path.join(results_dir, model_filename)
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[INFO] Saved {model_name} model to {model_filename}")

    prediction_rows = []
    for row in results_df.itertuples():
        model_name = row.Model
        model_instance = best_models[model_name]

        for subset_name, (X_sub, y_sub, fname_sub) in [
            ("Train", (X_train_df, y_train, fname_train)),
            ("Test", (X_test_df, y_test, fname_test))
        ]:
            preds = model_instance.predict(X_sub[numeric_cols])
            if fname_sub is not None:
                for actual, pred, fname in zip(y_sub, preds, fname_sub):
                    prediction_rows.append({
                        "Model": model_name,
                        "RandomState": row._2,
                        "Set": subset_name,
                        "Filename": fname,
                        "Actual": actual,
                        "Predicted": pred
                    })
            else:
                for actual, pred in zip(y_sub, preds):
                    prediction_rows.append({
                        "Model": model_name,
                        "RandomState": row._2,
                        "Set": subset_name,
                        "Filename": "",
                        "Actual": actual,
                        "Predicted": pred
                    })

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(os.path.join(results_dir, "predictions_each_model.csv"), index=False)
    print("[INFO] Predictions CSV created with filename labels (if present).")

    print("\n[INFO] Generating SHAP and permutation importance...")

    tree_based_models = ["Random Forest", "XGBoost"]

    for model_name, model in best_models.items():
        if model_name in tree_based_models:
            print(f"[INFO] Generating SHAP values for: {model_name}")

            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_test_df[numeric_cols])

            shap_df = pd.DataFrame({
                'Feature': numeric_cols,
                'MeanAbsSHAP': np.abs(shap_values.values).mean(axis=0)
            }).sort_values(by='MeanAbsSHAP', ascending=False)
            shap_df.to_csv(os.path.join(results_dir, f"{model_name.lower()}_shap_feature_importance.csv"), index=False)

            shap.summary_plot(shap_values, X_test_df[numeric_cols], plot_type="bar", show=False)
            plt.title(f"{model_name} - SHAP Feature Importance (Bar)")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{model_name}_shap_bar.png"))
            plt.clf()

            shap.summary_plot(shap_values, X_test_df[numeric_cols], show=False)
            plt.title(f"{model_name} - SHAP Summary Plot")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{model_name}_shap_summary.png"))
            plt.clf()

        print(f"[INFO] Generating permutation importance for: {model_name}")
        perm = permutation_importance(model, X_test_df[numeric_cols], y_test, n_repeats=10, random_state=42)
        perm_df = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance': perm.importances_mean
        }).sort_values(by='Importance', ascending=False)
        perm_df.to_csv(os.path.join(results_dir, f"{model_name.lower()}_permutation_importance.csv"), index=False)

    print("[INFO] Generating VIF analysis...")

    def compute_vif(df):
        vif_data = []
        for i in range(df.shape[1]):
            vif_value = variance_inflation_factor(df.values, i)
            vif_data.append({"Feature": df.columns[i], "VIF": vif_value})
        return pd.DataFrame(vif_data)

    vif_df = compute_vif(pd.DataFrame(X_train, columns=X.columns))
    vif_df.to_csv(os.path.join(results_dir, "vif_values.csv"), index=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(X_train, columns=X.columns).corr(), annot=True, cmap='viridis')
    plt.title("VIF Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "vif_correlation_heatmap.png"))
    plt.clf()

    print("[INFO] Saving visualizations...")

    y_test_pred = top_model.predict(X_test_df)

    plt.figure()
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    plt.xlabel('Actual Weight (lb)')
    plt.ylabel('Predicted Weight (lb)')
    plt.title('Actual vs. Predicted on Test Set')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "actual_vs_predicted.png"))
    plt.clf()

    residuals = y_test - y_test_pred
    plt.figure()
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle='--', color='red')
    plt.xlabel('Predicted Weight (lb)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "residual_plot.png"))
    plt.clf()

    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "prediction_error_histogram.png"))
    plt.clf()

    print("[INFO] Generating correlation matrix...")
    corr_matrix = df.drop(columns=drop_cols).corr()
    corr_matrix.to_csv(os.path.join(results_dir, "correlation_matrix.csv"))

    plt.figure(figsize=(24, 20))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="viridis", square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "correlation_heatmap.png"))
    plt.clf()

    readme_path = os.path.join(results_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Sorghum Yield Prediction Pipeline\n")
        f.write(f"**Generated on:** {timestamp}\n\n")
        f.write("This folder contains the output from a machine learning pipeline built to predict sorghum yield (`weight_lb`).\n\n")
        f.write("## Models Trained\n- Ridge\n- Lasso\n- ElasticNet\n- Random Forest\n- XGBoost (GPU)\n- SVR\n\n")
        f.write("## Files Included\n")
        f.write("- model_selection_summary.csv\n")
        f.write("- predictions_each_model.csv\n")
        f.write("- shap_feature_importance.csv\n")
        f.write("- shap_values_matrix.csv\n")
        f.write("- permutation_importance.csv\n")
        f.write("- correlation_matrix.csv\n")
        f.write("- vif_values.csv\n")
        f.write("- vif_correlation_heatmap.png\n")
        f.write("- PNG visualizations (SHAP, residuals, predictions, correlation heatmap)\n")
        f.write("- Best model pickle files (.pkl) for each algorithm\n")
        f.write("- environment.json\n\n")
        f.write("## How to Reproduce\nRun the script (`<script_name>.py`) inside a compatible Python environment with the required libraries installed.\n")

    pdf_path = os.path.join(results_dir, "model_summary_report.pdf")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Sorghum Yield Prediction Summary", align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated: {timestamp}")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 10, "Test Set Performance Summary (Sorted by Best R²):")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 9)
    pdf.cell(35, 7, "Model", border=1)
    pdf.cell(20, 7, "Best RS", border=1)
    pdf.cell(25, 7, "Best R2", border=1)
    pdf.cell(25, 7, "Best RMSE", border=1)
    pdf.cell(40, 7, "Mean R2 (± Std)", border=1)
    pdf.cell(45, 7, "Mean RMSE (± Std)", border=1)
    pdf.ln()

    pdf.set_font("Arial", "", 9)
    for _, row in results_df.iterrows():
        mean_r2_std = f"{row['Mean R2']:.3f} \u00B1 {row['Std R2']:.3f}"
        mean_rmse_std = f"{row['Mean RMSE']:.3f} \u00B1 {row['Std RMSE']:.3f}"
        pdf.cell(35, 7, row['Model'], border=1)
        pdf.cell(20, 7, str(row['Best Random State']), border=1, align='C')
        pdf.cell(25, 7, f"{row['Best Test R2']:.3f}", border=1, align='R')
        pdf.cell(25, 7, f"{row['Best Test RMSE']:.3f}", border=1, align='R')
        pdf.cell(40, 7, mean_r2_std, border=1, align='R')
        pdf.cell(45, 7, mean_rmse_std, border=1, align='R')
        pdf.ln()

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 10, "Variance Inflation Factor (VIF) Results:")
    pdf.ln(5)
    pdf.set_font("Arial", "", 10)
    for _, row in vif_df.sort_values(by="VIF", ascending=False).iterrows():
        pdf.multi_cell(0, 6, f"{row['Feature']}: {row['VIF']:.3f}")

    pdf.ln(5)
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 10, "Visualizations")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 11)
    pdf.multi_cell(0, 8, "VIF Correlation Heatmap:")
    pdf.ln(2)
    vif_heatmap_path = os.path.join(results_dir, "vif_correlation_heatmap.png")
    if os.path.exists(vif_heatmap_path):
        pdf.image(vif_heatmap_path, w=170)
    pdf.ln(5)

    pngs = [
        ("Actual vs Predicted (Test Set)", "actual_vs_predicted.png"),
        ("Residual Plot (Test Set)", "residual_plot.png"),
        ("Prediction Error Distribution (Test Set)", "prediction_error_histogram.png"),
        ("Feature Correlation Matrix (Original Data)", "correlation_heatmap.png")
    ]

    if top_model_name in tree_based_models:
        pngs.insert(3, (f"{top_model_name} - SHAP Importance (Bar)", f"{top_model_name.lower()}_shap_bar.png"))
        pngs.insert(4, (f"{top_model_name} - SHAP Summary Plot", f"{top_model_name.lower()}_shap_summary.png"))

    for title, png_filename in pngs:
        path = os.path.join(results_dir, png_filename)
        if os.path.exists(path):
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(0, 8, f"{title}:")
            pdf.ln(2)
            pdf.image(path, w=170)
            pdf.ln(5)
        else:
            pdf.set_font("Arial", "I", 10)
            pdf.multi_cell(0, 8, f"({title} image not found at {path})")
            pdf.ln(5)

    pdf.output(pdf_path, "F")
    print("[INFO] README.md and summary PDF generated!")
    print("[INFO] All done! 🎉")
    print(f"Results saved to: {results_dir}")
