import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.impute import KNNImputer


# --- Data Transformation Functions ---
def is_convertible_to_float(input_val):
    try:
        if input_val is None or pd.isna(input_val): return False
        float(input_val)
        return True
    except (ValueError, TypeError): return False

def convert_value(df, col_name, row_idx, mean_morning_type_val, mean_mouth_opening_val):
    try: val = df.loc[row_idx, col_name]
    except (KeyError, IndexError): return
    converted_val = val
    if col_name == "Bilateral pain ": converted_val = 0 if pd.isna(val) else 3
    elif col_name in ["Bruxism","Clenching", "Psychological stress","Tinnitus"]:
        if pd.isna(val): converted_val = 0
        elif isinstance(val, str) and val.strip().startswith("1"): converted_val = 1
        elif is_convertible_to_float(val): converted_val = float(val)
        else: converted_val = 0
    elif col_name == "Morning type": converted_val = mean_morning_type_val if pd.isna(val) else (float(val) if is_convertible_to_float(val) else mean_morning_type_val)
    elif isinstance(col_name, str) and (col_name.startswith("PSQI") or col_name == "Poor sleeper"):
        psqi_base_cols_for_avg = [f"PSQI{i}" for i in range(1, 8)]
        sum_psqi, cnt_psqi = 0, 0
        for k_col_name in psqi_base_cols_for_avg:
            if k_col_name in df.columns:
                 try:
                     k_val = df.loc[row_idx, k_col_name]
                     if is_convertible_to_float(k_val):
                         sum_psqi += float(k_val)
                         cnt_psqi += 1
                 except (KeyError, IndexError): continue
        if col_name == "PSQI score": converted_val = sum_psqi
        elif col_name == "Poor sleeper": converted_val = int(sum_psqi >= 6)
        elif col_name.startswith("PSQI") and col_name != "PSQI score":
             if is_convertible_to_float(val): converted_val = float(val)
             elif cnt_psqi > 0: converted_val = sum_psqi / cnt_psqi
             else: converted_val = 0
    elif col_name == "Mouth opening ": converted_val = mean_mouth_opening_val if pd.isna(val) else (0 if str(val).strip() == "없음" else 1)
    else:
        if not is_convertible_to_float(val): converted_val = 0
        elif isinstance(val, (str, bool, int)):
             try: converted_val = float(val)
             except ValueError: converted_val = 0
        elif isinstance(val, (float, np.number)): converted_val = val
    try: df.loc[row_idx, col_name] = converted_val
    except Exception: pass

def calculate_columns_mean(z_col, data):
    mean_z = 0
    if z_col in data.columns:
        temp_z_series = pd.to_numeric(data[z_col], errors='coerce')
        negative_values = temp_z_series[temp_z_series < 0]
        if not negative_values.empty:
            sum_z = negative_values.sum()
            cnt_z = negative_values.count()
            mean_z = sum_z / cnt_z if cnt_z > 0 else 0
    return mean_z

def get_features():
    # --- Define Feature Groups ---
    target_columns = ["VAS", "CMI"]
    excluded_features = ["DI", "PI"]
    patient_report_features = ["CMI_Patient", "DI_Patient", "PI_Patient"]
    sleep_related_features = ["PSQI1", "PSQI2", "PSQI3", "PSQI4", "PSQI5", "PSQI6", "PSQI7", "PSQI score", "Poor sleeper",
                            "STOP-Bang1", "STOP-Bang2", "STOP-Bang3", "STOP-Bang4", "STOP-Bang5", "STOP-Bang6", "STOP-Bang7", "STOP-Bang8", "STOP-Bang score", "High risk for OSA"]
    return target_columns, excluded_features, patient_report_features, sleep_related_features

def preprocessing_data():
    # --- Data Loading and Preprocessing ---
    print("\n--- Loading and Preprocessing Data ---")
    try:
        pure_data = pd.read_excel("data/dataset.xlsx", sheet_name="sheet1")
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: dataset.xlsx not found.")
        exit()
    pure_data = pure_data.iloc[2:]

    try:
        if 'Study date ' in pure_data.columns:
            pure_data['Study date '] = pd.to_datetime(pure_data['Study date '], errors='coerce')
        else:
            print("Warning: 'Study date' column not found.")
    except Exception as e:
        print(f"Warning: Could not parse 'Study date ': {e}")

    if 'Study date ' in pure_data.columns and pd.api.types.is_datetime64_any_dtype(pure_data['Study date ']):
        df_sorted = pure_data.sort_values(by='Study date ')
        print("Data sorted by 'Study date '.")
    else:
        df_sorted = pure_data
        print("Warning: Not sorting by 'Study date '.")


    target_columns, _, _, _ = get_features()

    try:
        all_cols_in_range = df_sorted.columns[4:46].tolist()
        potential_feature_cols = [col for col in all_cols_in_range if col not in target_columns]
        data_raw = df_sorted[potential_feature_cols].copy()
        target_raw = df_sorted[target_columns].copy()

    except (IndexError, KeyError) as e:
        print(f"Error selecting initial columns: {e}")
        exit()

    data = data_raw.reset_index(drop=True)
    target_df = target_raw.reset_index(drop=True)



    mean_morning_type = calculate_columns_mean("Morning type", data)
    mean_mouth_opening = calculate_columns_mean("Mouth opening ", data)

    object_columns = data.select_dtypes(include=['object']).columns.tolist()
    special_numeric_cols = ["Bilateral pain ", "Morning type", "Mouth opening "]
    psqi_related_cols = [c for c in data.columns if isinstance(c, str) and (c.startswith("PSQI") or c == "Poor sleeper")]
    contributing_factor_cols = ["Bruxism","Clenching", "Psychological stress","Tinnitus"]
    cols_to_convert = list(set(object_columns + special_numeric_cols + psqi_related_cols + contributing_factor_cols))

    for col in tqdm(data.columns, desc="Converting Columns"):
        if col in cols_to_convert:
            for row_idx in range(len(data)):
                convert_value(data, col, row_idx, mean_morning_type, mean_mouth_opening)

    data = data.apply(pd.to_numeric, errors='coerce')
    target_df = target_df.apply(pd.to_numeric, errors='coerce')

    corr_matrix = data.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
    to_drop_corr = [column for column in upper_tri.columns if upper_tri[column].gt(0.95).any()]
    to_drop = list(set(to_drop_corr))
    id_col = "ID.1"
    if id_col in data.columns and id_col not in to_drop: to_drop.append(id_col)
    if to_drop:
        print(f"  Columns to drop (High Corr / ID): {to_drop}")
        data = data.drop(columns=to_drop, errors='ignore')

    combined_df = pd.concat([data, target_df], axis=1)
    all_nan_cols = combined_df.columns[combined_df.isna().all()].tolist()
    if all_nan_cols:
        print(f"Warning: Dropping all-NaN columns: {all_nan_cols}")
        combined_df = combined_df.drop(columns=all_nan_cols)
        target_columns = [col for col in target_columns if col not in all_nan_cols]

    imputer = KNNImputer(n_neighbors=5)
    combined_df_imputed = pd.DataFrame(imputer.fit_transform(combined_df), columns=combined_df.columns)

    final_target_names = [col for col in target_columns if col in combined_df_imputed.columns]

    return combined_df_imputed, final_target_names