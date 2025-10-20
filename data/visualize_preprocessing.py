import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import seaborn as sns

from sklearn.model_selection import StratifiedKFold
import warnings
from data_preprocessing import preprocessing_data, get_features

warnings.filterwarnings('ignore', category=FutureWarning)

def map_psqi3_to_category_label(psqi3_value):
    if pd.isna(psqi3_value): return 'Unknown'
    psqi3_int = int(psqi3_value)
    if psqi3_int == 0: return '≥7h'
    elif psqi3_int == 1: return '6-7h'
    elif psqi3_int == 2: return '5-6h'
    elif psqi3_int == 3: return '<5h'
    else: return 'Unknown'

def plot_psqi3_vs_pain_violin_point(df, filename_cmi="PSQI3_vs_CMI_ViolinPoint.png", filename_vas="PSQI3_vs_VAS_ViolinPoint.png"):
    if 'PSQI3' not in df.columns or 'CMI' not in df.columns or 'VAS' not in df.columns:
        print("Error: Required columns ('PSQI3', 'CMI', 'VAS') not found.")
        return

    df['PSQI3_numeric'] = pd.to_numeric(df['PSQI3'], errors='coerce')
    df_plot = df.dropna(subset=['PSQI3_numeric', 'CMI', 'VAS']).copy()
    df_plot['PSQI3_numeric'] = df_plot['PSQI3_numeric'].astype(int)
    df_plot['Sleep Duration Label'] = df_plot['PSQI3_numeric'].apply(map_psqi3_to_category_label)
    category_order = ['≥7h', '6-7h', '5-6h', '<5h']
    valid_order = [cat for cat in category_order if cat in df_plot['Sleep Duration Label'].unique()]
    df_plot['Sleep Duration Label'] = pd.Categorical(df_plot['Sleep Duration Label'], categories=valid_order, ordered=True)

    if df_plot.empty:
        print("No valid data points after handling PSQI3."); return

    print(f"Generating Violin + Point plot (Pastel) for PSQI3 vs. CMI...")
    fig_cmi, ax_cmi = plt.subplots(figsize=(10, 7))
    fig_cmi.suptitle('(A) Sleep Duration vs. CMI', fontsize=16, y=0.98)

    sns.violinplot(x='Sleep Duration Label', y='CMI', data=df_plot, ax=ax_cmi, palette='GnBu',
                   order=valid_order, inner=None, linewidth=1.5, cut=0, alpha=0.7)
    sns.pointplot(x='Sleep Duration Label', y='CMI', data=df_plot, ax=ax_cmi,
                  order=valid_order, color='grey', markers='o', linestyles='-',
                  errorbar='ci', capsize=.3, label='Mean Score Trend', alpha=0.7)
    sns.stripplot(x='Sleep Duration Label', y='CMI', data=df_plot, ax=ax_cmi, color='lightgrey',
                  alpha=0.1, jitter=0.25, order=valid_order, s=3)

    ax_cmi.set_title('')
    ax_cmi.set_xlabel('Sleep Duration Category (from PSQI3)')
    ax_cmi.set_ylabel('CMI Score (Clinician-Assessed)')
    ax_cmi.tick_params(axis='x')
    handles_cmi, labels_cmi = ax_cmi.get_legend_handles_labels()
    pointplot_handles_cmi = [h for h, l in zip(handles_cmi, labels_cmi) if isinstance(h, Line2D) and h.get_color() == 'grey']
    if pointplot_handles_cmi:
         ax_cmi.legend(handles=pointplot_handles_cmi, labels=['Mean Score Trend'], loc='upper right')
    ax_cmi.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(filename_cmi, dpi=300)
    plt.close(fig_cmi)
    print(f"CMI Plot saved as '{filename_cmi}'")

    print(f"Generating Violin + Point plot (Pastel) for PSQI3 vs. VAS...")
    fig_vas, ax_vas = plt.subplots(figsize=(10, 7))
    fig_vas.suptitle('(B) Sleep Duration vs. VAS', fontsize=16, y=0.98)

    sns.violinplot(x='Sleep Duration Label', y='VAS', data=df_plot, ax=ax_vas, palette='OrRd',
                   order=valid_order, inner=None, linewidth=1.5, cut=0, alpha=0.7)
    sns.pointplot(x='Sleep Duration Label', y='VAS', data=df_plot, ax=ax_vas,
                  order=valid_order, color='grey', markers='o', linestyles='-',
                  errorbar='ci', capsize=.3, label='Mean Score Trend', alpha=0.7)
    sns.stripplot(x='Sleep Duration Label', y='VAS', data=df_plot, ax=ax_vas, color='lightgrey',
                  alpha=0.1, jitter=0.25, order=valid_order, s=3)

    ax_vas.set_title('')
    ax_vas.set_xlabel('Sleep Duration Category (from PSQI3)')
    ax_vas.set_ylabel('VAS Score (Patient-Reported)')
    ax_vas.tick_params(axis='x')
    handles_vas, labels_vas = ax_vas.get_legend_handles_labels()
    pointplot_handles_vas = [h for h, l in zip(handles_vas, labels_vas) if isinstance(h, Line2D) and h.get_color() == 'grey']
    if pointplot_handles_vas:
         ax_vas.legend(handles=pointplot_handles_vas, labels=['Mean Score Trend'], loc='upper right')
    ax_vas.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(filename_vas, dpi=300)
    plt.close(fig_vas)
    print(f"VAS Plot saved as '{filename_vas}'")

def create_stratified_cv_plot(df, target_col, n_splits=10, filename='stratified_cv_plot.png'):
    print(f"Generating corrected Stratified CV plot for '{target_col}'...")
    
    df_sorted = df.dropna(subset=[target_col]).sort_values(by=target_col).reset_index(drop=True)
    y = df_sorted[target_col]
    X = df_sorted.drop(columns=[target_col])
    bins = pd.qcut(y, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = {
        'Low': '#A1C9F4', 
        'Medium': '#F8B582', 
        'High': '#8DE5A1',
        'Training Set': '#DCDCDC',
        'Test Set': '#F89F9B'
    }

    group_counts = bins.value_counts().sort_index()
    bottom = 0
    for group_name, count in group_counts.items():
        ax.bar(0, count, bottom=bottom, color=colors[group_name], edgecolor='black', width=0.8)
        bottom += count

    for i, (train_idx, test_idx) in enumerate(skf.split(X, bins)):
        ax.bar(i + 1, len(df_sorted), color=colors['Training Set'], edgecolor='black', width=0.8)
        
        test_idx_sorted = np.sort(test_idx)
        split_indices = np.where(np.diff(test_idx_sorted) != 1)[0] + 1
        contiguous_blocks = np.split(test_idx_sorted, split_indices)
        
        for block in contiguous_blocks:
            if block.size > 0:
                start_index = block[0]
                height = len(block)
                ax.bar(i + 1, height, bottom=start_index, color=colors['Test Set'], edgecolor='black', width=0.8, linewidth=0.5)

    group_boundaries = group_counts.cumsum()
    ax.axhline(y=group_boundaries.iloc[0], color='black', linestyle='--', dashes=(5, 5), zorder=3)
    ax.axhline(y=group_boundaries.iloc[1], color='black', linestyle='--', dashes=(5, 5), zorder=3)
    
    ax.set_xticks(range(n_splits + 1))
    xticklabels = [f'{target_col} Groups'] + [f'Fold {i+1}' for i in range(n_splits)]
    
    ax.set_xticklabels(xticklabels, rotation=30, ha='right')
    
    ax.set_ylabel(f'Sample Index (Sorted by {target_col} Group)')
    ax.set_title(f'Stratified {n_splits}-Fold Cross-Validation for {target_col} Variable', fontsize=14)
    
    legend_patches = [
        mpatches.Patch(color=colors['Low'], label=f'Low {target_col}'),
        mpatches.Patch(color=colors['Medium'], label=f'Medium {target_col}'),
        mpatches.Patch(color=colors['High'], label=f'High {target_col}'),
        mpatches.Patch(color=colors['Training Set'], label='Training Set'),
        mpatches.Patch(color=colors['Test Set'], label='Test Set')
    ]
    
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot correctly saved as '{filename}'\n")


def create_distribution_comparison_plot(df_before, df_after, col_name, filename='dist_comparison.png'):
    print(f"Generating distribution comparison plot for '{col_name}'...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df_before[col_name].dropna(), color="#a2d5ab", label="Before Processing", kde=True, stat='density', alpha=0.6)
    sns.histplot(df_after[col_name].dropna(), color="#c3b1e1", label="After Processing", kde=True, stat='density', alpha=0.6)
    plt.title(f"Distribution of '{col_name}' Before vs. After Processing", fontsize=14)
    plt.xlabel(col_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as '{filename}'\n")

def create_correlation_heatmap(df, filename='corr_heatmap.png'):
    print("Generating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    
    corr_matrix = df.corr()
    
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.1f',
                annot_kws={"size": 7}, cbar_kws={"shrink": .8},
                square=True, linewidths=.5)

    plt.title('Feature Correlation After Selection', fontsize=14) # 제목 폰트 크기 조정
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as '{filename}'\n")



def generate_summary_table(df):
    
    summary_list = []
    
    for col in df.columns:
        if df[col].nunique() > 10 and pd.api.types.is_numeric_dtype(df[col].dtype):
            mean = df[col].mean()
            std = df[col].std()
            summary_list.append({
                'Variable': col.strip(),
                'Category': '',
                'Value': f'{mean:.2f} ± {std:.2f}'
            })
        else:
            s = df[col].astype(int)
            counts = s.value_counts().sort_index()
            percentages = s.value_counts(normalize=True).sort_index() * 100
            
            is_first = True
            for category, count in counts.items():
                percentage = percentages[category]
                summary_list.append({
                    'Variable': col.strip() if is_first else '',
                    'Category': category,
                    'Value': f'{count:,} ({percentage:.1f}%)'
                })
                is_first = False
                
    summary_df = pd.DataFrame(summary_list)
    print("\n--- Descriptive Statistics Table (Table 1) ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    file_path = './dataset.xlsx'
    
    print("\n--- Loading and Preprocessing Data ---")
    pure_data = pd.read_excel(file_path, sheet_name="sheet1")
    print("Dataset loaded successfully.")
    pure_data = pure_data.iloc[2:]

    if 'Study date ' in pure_data.columns:
        pure_data['Study date '] = pd.to_datetime(pure_data['Study date '], errors='coerce')
    else:
        print("Warning: 'Study date' column not found.")
    
    if 'Study date ' in pure_data.columns and pd.api.types.is_datetime64_any_dtype(pure_data['Study date ']):
        df_sorted = pure_data.sort_values(by='Study date ')
        print("Data sorted by 'Study date '.")
    else:
        df_sorted = pure_data
        print("Warning: Not sorting by 'Study date '.")


    target_columns, _, _, _ = get_features()

    all_cols_in_range = df_sorted.columns[4:46].tolist()
    potential_feature_cols = [col for col in all_cols_in_range if col not in target_columns]
    data_raw = df_sorted[potential_feature_cols].copy()
    
    for col in data_raw.select_dtypes(include=['object']).columns:
        data_raw[col] = pd.to_numeric(data_raw[col], errors='coerce')
    
    target_raw = df_sorted[target_columns].copy()
    df_raw = pd.concat([data_raw, target_raw], axis=1)
    
    df_processed, _ = preprocessing_data(file_path=file_path)
    
    if df_raw is not None and df_processed is not None:
        create_stratified_cv_plot(df_processed, 'CMI', filename='figure_1_stratified_cv_cmi.png')
        create_stratified_cv_plot(df_processed, 'VAS', filename='figure_2_stratified_cv_vas.png')
        
        if 'PSQI score' in df_raw.columns and 'PSQI score' in df_processed.columns:
            create_distribution_comparison_plot(df_raw, df_processed, 'PSQI score', filename='figure_3_dist_psqi.png')
        else:
            print("Skipping 'PSQI score' plot: column not found.")

        if 'CMI_Patient' in df_raw.columns and 'CMI_Patient' in df_processed.columns:
             create_distribution_comparison_plot(df_raw, df_processed, 'CMI_Patient', filename='figure_4_dist_cmi_patient.png')
        else:
            print("Skipping 'CMI_Patient' plot: column not found.")

        morning_type_col = 'Morning person' 
        if morning_type_col in df_raw.columns and morning_type_col in df_processed.columns:
            create_distribution_comparison_plot(df_raw, df_processed, morning_type_col, filename='figure_5_dist_morning_person.png')
        else:
            print(f"Skipping '{morning_type_col}' plot: column not found. Please check the exact column name in your file.")
        create_distribution_comparison_plot(df_raw, df_processed, 'VAS', filename='figure_5b_dist_vas.png')
        create_distribution_comparison_plot(df_raw, df_processed, 'CMI', filename='figure_5c_dist_cmi.png')
        
        
        create_correlation_heatmap(df_processed, filename='figure_6_corr_heatmap.png')

        print("\nAll plots have been generated and saved successfully.")
        
        generate_summary_table(df_processed)
        
        plot_psqi3_vs_pain_violin_point(df_processed)