import torch
import numpy as np
import json
import math
from tqdm import tqdm
from sklearn.model_selection import KFold

from utils.estimate_MI_score import KL_div_ks_torch, MI_from_divergence_v2


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# --- Standardization Function ---
def standardize(data, eps=1e-8):
    if not isinstance(data, torch.Tensor):
        raise TypeError("Input data must be a PyTorch Tensor")
    if data.numel() == 0:
        return data
    original_ndim = data.ndim
    if original_ndim == 1:
        mean = torch.mean(data)
        std = torch.std(data)
        standardized_data = (data - mean) / (std + eps)
    elif original_ndim == 2:
        mean = torch.mean(data, dim=0, keepdim=True)
        std = torch.std(data, dim=0, keepdim=True)
        mean = torch.nan_to_num(mean, nan=0.0)
        std = torch.nan_to_num(std, nan=1.0)
        standardized_data = (data - mean) / (std + eps)
    else:
        raise ValueError("Standardization only supports 1D or 2D tensors.")
    if torch.isnan(standardized_data).any() or torch.isinf(standardized_data).any():
        standardized_data = torch.nan_to_num(standardized_data, nan=0.0, posinf=1e6, neginf=-1e6)
    return standardized_data


# --- Helper Functions (Feature Selection, Model Training) ---
def calculate_mi_once(feature_2d, y_scaled, **kwargs):
    try:
        mi_val = kwargs['mi_func'](feature_2d, y_scaled, **kwargs).item()
        return mi_val if not (np.isnan(mi_val) or np.isinf(mi_val)) else -np.inf
    except Exception:
        return -np.inf

def select_features_mi_with_permutation_test(X, y, feature_names, n_permutations=100, **kwargs_for_mi):
    effective_device_mi = kwargs_for_mi.get('device', torch.device('cpu'))
    print(f"\n--- Starting MI Calculation with Permutation Test ({n_permutations} iterations) ---")
    X, y = X.to(effective_device_mi), y.to(effective_device_mi)
    X_scaled, y_scaled = standardize(X), standardize(y)
    if y_scaled.ndim == 1: y_scaled = y_scaled.unsqueeze(1)
    mi_scores_list = []
    for i in tqdm(range(X.shape[1]), desc="Calculating MI and P-values"):
        feature_2d = X_scaled[:, i].unsqueeze(1)
        original_mi_scores = [calculate_mi_once(feature_2d, y_scaled, **kwargs_for_mi) for _ in range(10)]
        original_mi = np.mean([s for s in original_mi_scores if s > -np.inf])
        null_mi_scores = []
        for _ in range(n_permutations):
            y_shuffled = y_scaled[torch.randperm(y_scaled.shape[0])]
            null_mi = calculate_mi_once(feature_2d, y_shuffled, **kwargs_for_mi)
            if null_mi > -np.inf: null_mi_scores.append(null_mi)
        p_value = (np.sum(np.array(null_mi_scores) >= original_mi) + 1) / (len(null_mi_scores) + 1)
        mi_scores_list.append((feature_names[i], original_mi, p_value))
    ranked_features = sorted(mi_scores_list, key=lambda x: x[1], reverse=True)
    return ranked_features

def evaluate_model_with_kfold(X, y, target_names, n_splits=10, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results_per_target = {name: {'mse_scores': [], 'rmse_scores': []} for name in target_names}
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler().fit(X_train)
        X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)
        for i, name in enumerate(target_names):
            model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
            model.fit(X_train_scaled, y_train[:, i])
            preds = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test[:, i], preds)
            results_per_target[name]['mse_scores'].append(mse)
            results_per_target[name]['rmse_scores'].append(math.sqrt(mse))
    final_results = {}
    for name, scores_dict in results_per_target.items():
        final_results[name] = {
            'mse': np.mean(scores_dict['mse_scores']), 'rmse': np.mean(scores_dict['rmse_scores']),
            'mse_std': np.std(scores_dict['mse_scores']), 'rmse_std': np.std(scores_dict['rmse_scores'])
        }
    return final_results


def run_analysis_pipeline(data_df, feature_columns_to_use, target_columns_list, scenario_prefix, device, json_file_name=None):
    print(f"\n{'='*30}\nRunning pipeline for: {scenario_prefix}\n{'='*30}")
    data_for_model_pd = data_df[feature_columns_to_use]
    targets_for_model_pd = data_df[target_columns_list]
    features_tensor = torch.tensor(data_for_model_pd.values, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_for_model_pd.values, dtype=torch.float32)
    
    mi_params = {'mi_func': MI_from_divergence_v2, 'divergence_func': KL_div_ks_torch, 'n_permute': 5, 'min_k': 7, 'max_k': 15, 'Nmid_k': 1, 'run_gpu': -1, 'device': device}
    ranked_mi_scores = select_features_mi_with_permutation_test(features_tensor, targets_tensor, feature_columns_to_use, n_permutations=1000, **mi_params)
    
    print(f"\n--- {scenario_prefix} - Ranked MI Scores with P-values ---")
    for name, score, p_value in ranked_mi_scores:
        significance = "*" if p_value < 0.05 else ""
        print(f"{name}: MI={score:.6f}, p={p_value:.4f} {significance}")
    
    all_model_results = {}
    X_all, Y_all = data_for_model_pd.values, targets_for_model_pd.values

    # 시나리오 1: 모든 변수
    print(f"\n--- {scenario_prefix} - Scenario: All Available Features ---")
    all_model_results['All Features'] = evaluate_model_with_kfold(X_all, Y_all, target_columns_list)

    # 시나리오 2: MI 상위 20개 변수
    print(f"\n--- {scenario_prefix} - Scenario: MI Top 20 Features ---")
    valid_ranked_features = [item for item in ranked_mi_scores if not np.isinf(item[1]) and not np.isnan(item[1])]
    if len(valid_ranked_features) >= 20:
        top_20_features = [name for name, _, _ in valid_ranked_features[:20]]
        X_top20 = data_for_model_pd[top_20_features].values
        all_model_results['MI Top 20'] = evaluate_model_with_kfold(X_top20, Y_all, target_columns_list)
    else:
        all_model_results['MI Top 20'] = {}
        
    # [새로운 시나리오] 시나리오 3: 유의미한 MI 변수 (p < 0.05)
    print(f"\n--- {scenario_prefix} - Scenario: Significant MI Features (p < 0.05) ---")
    significant_features = [name for name, _, p_value in ranked_mi_scores if p_value < 0.05]
    if significant_features:
        print(f"  Found {len(significant_features)} significant features: {significant_features}")
        X_sig = data_for_model_pd[significant_features].values
        all_model_results['Significant MI Features'] = evaluate_model_with_kfold(X_sig, Y_all, target_columns_list)
    else:
        print("  No significant features found. Skipping this scenario.")
        all_model_results['Significant MI Features'] = {}

    print(all_model_results)
    print(ranked_mi_scores)
    
    if json_file_name:
        file_name = json_file_name
    else:
        file_name = f"{scenario_prefix}_model_results.json"
    with open(file_name, 'w') as f:
        json.dump(all_model_results, f, indent=4)
    print(f"Results saved to {file_name}")
    
    return all_model_results, ranked_mi_scores

    