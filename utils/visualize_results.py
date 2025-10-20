import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

import seaborn as sns
from matplotlib.lines import Line2D

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

PALETTE_LONG_NAME = {
    'Full_Model': '#95B1D2',
    'Clinical_Sleep_Model': '#E7B43C',
    'Clinical_Only_Model': '#E16D94'
}

def load_all_performance_data(model_names):
    all_plot_data = []
    for model_name_full in model_names:
        filename = f"./{model_name_full}_model_results.json"
        try:
            with open(filename, 'r') as f:
                all_scenarios_results = json.load(f)
            for scenario_name, scenario_results in all_scenarios_results.items():
                if not scenario_results: continue
                for target, metrics in scenario_results.items():
                    for metric_key in ['mse_scores', 'rmse_scores', 'r2_scores']:
                        if metric_key in metrics and metrics[metric_key] is not None:
                            metric_name = metric_key.replace('_scores', '').upper()
                            for score in metrics[metric_key]:
                                all_plot_data.append({
                                    'Model': model_name_full.replace('_Model', '').replace('_', ' '),
                                    'Scenario': scenario_name, 'Target': target,
                                    'Metric': metric_name, 'Value': score
                                })
        except FileNotFoundError:
            print(f"Warning: Result file not found: '{filename}'")
    return pd.DataFrame(all_plot_data)

def load_all_mi_data(model_names):
    all_mi_data = []
    for model_name in model_names:
        filename = f"./{model_name}_mi_scores.json"
        try:
            with open(filename, 'r') as f:
                df = pd.DataFrame(json.load(f))
            df['Model'] = model_name
            all_mi_data.append(df)
        except FileNotFoundError:
            print(f"Warning: MI score file not found: '{filename}'")
    if not all_mi_data: return pd.DataFrame()
    return pd.concat(all_mi_data, ignore_index=True)

PALETTE = {
    'Full': "#95B1D2",
    'Clinical Sleep': "#E7B43C",
    'Clinical Only': "#E16D94"
}


def load_p_values_from_json(filepath):
    
    print(f"Loading p-values from '{filepath}'...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        # 그룹 순서를 일관되게 정렬하여 로드
        for item in data:
            g1, g2 = sorted([item['Group1'], item['Group2']])
            item['Group1'], item['Group2'] = g1, g2
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: P-value file not found at '{filepath}'")
        return pd.DataFrame()

def format_p_value(p):
    if p < 0.001: return "p < 0.001"
    if p < 0.01: return "p < 0.01"
    return f"p = {p:.3f}"

def draw_p_value_bracket(ax, p_value_text, x1, x2, y_start, bar_height_fraction=0.02):
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bar_height = y_range * bar_height_fraction
    
    ax.plot([x1, x1, x2, x2], [y_start, y_start + bar_height, y_start + bar_height, y_start], c='black', lw=1.5)    
    ax.text((x1 + x2) * 0.5, y_start + bar_height, p_value_text, ha='center', va='bottom', color='black', fontsize=12)

def plot_performance_comparison(performance_df, p_values_df, scenario_title, filename):

    print(f"Generating performance plot for scenario: {scenario_title}...")
    scenario_df = performance_df[performance_df['Scenario'] == scenario_title]
    if scenario_df.empty:
        print(f"No data for scenario '{scenario_title}'. Skipping plot."); return

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    fig.suptitle(f'Model Performance Comparison ({scenario_title})', fontsize=20, y=0.98)
    
    metrics = ['MSE', 'RMSE', 'R2']; targets = ['CMI', 'VAS']
    model_order = ['Full', 'Clinical Sleep', 'Clinical Only']

    for i, metric in enumerate(metrics):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            subset = scenario_df[(scenario_df['Metric'] == metric) & (scenario_df['Target'] == target)]
            
            sns.boxplot(x='Model', y='Value', data=subset, order=model_order,
                        hue='Model', palette=PALETTE, ax=ax, width=0.6, legend=False,
                        boxprops=dict(edgecolor='black', alpha=0.8), medianprops=dict(color='black'),
                        whiskerprops=dict(color='black', alpha=0.8), capprops=dict(color='black', alpha=0.8))
            
            sns.stripplot(x='Model', y='Value', data=subset, order=model_order,
                          hue='Model', palette=PALETTE, ax=ax, alpha=0.8, dodge=True,
                          edgecolor='black', linewidth=0.5, legend=False)
            
            ax.set_title(f'Target: {target} | Metric: {metric}', fontsize=14)
            ax.set_ylabel('Metric Value'); ax.set_xlabel('')

            if i < len(metrics) - 1:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='x', rotation=30)
            
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

            if not subset.empty:
                y_max = subset['Value'].max()
                y_min = subset['Value'].min()
                y_range = y_max - y_min if y_max > y_min else 1.0

                p_values_subset = p_values_df[(p_values_df['Metric'] == metric) & 
                                              (p_values_df['Target'] == target) &
                                              (p_values_df['Scenario'] == scenario_title)]
                
                pairs_map = {
                    tuple(sorted(['Full', 'Clinical Sleep'])): {'pos': [0, 1], 'offset': 0.10},
                    tuple(sorted(['Full', 'Clinical Only'])): {'pos': [0, 2], 'offset': 0.30},
                    tuple(sorted(['Clinical Sleep', 'Clinical Only'])): {'pos': [1, 2], 'offset': 0.20}
                }
                
                for _, row in p_values_subset.iterrows():
                    key = tuple(sorted([row['Group1'], row['Group2']]))
                    if key in pairs_map:
                        p_text = format_p_value(row['p-value'])
                        pos = pairs_map[key]['pos']
                        y_pos = y_max + (y_range * pairs_map[key]['offset'])
                        draw_p_value_bracket(ax, p_text, pos[0], pos[1], y_pos)

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as '{filename}'")
    
def plot_volcano(mi_df, filename):
    print("Generating Volcano Plot...")
    if mi_df.empty:
        print("Skipping Volcano Plot due to missing MI data.")
        return

    mi_df['-log10(p_value)'] = -np.log10(mi_df['p_value'].replace(0, 1e-300))

    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=mi_df, x='mi_score', y='-log10(p_value)', 
                         hue='Model', style='Model', palette=PALETTE_LONG_NAME, s=100, alpha=0.8)
    
    ax.axhline(y=-np.log10(0.05), color='grey', linestyle='--')
    sns.regplot(data=mi_df, x='mi_score', y='-log10(p_value)', 
                scatter=False, lowess=True, ax=ax, 
                line_kws={'color': 'grey', 'linestyle': ':'})

    features_to_label = mi_df[(mi_df['p_value'] < 0.05) & (mi_df['mi_score'] > 0.2)]

    texts = []
    for i, row in features_to_label.iterrows():
        texts.append(ax.text(row['mi_score'], row['-log10(p_value)'], row['feature'], fontsize=9))

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        
    ax.set_title('Feature Significance and Informativeness (Volcano Plot)', fontsize=16)
    ax.set_xlabel('MI Score')
    ax.set_ylabel('-log10(p_value)')
    
    handles, labels = ax.get_legend_handles_labels()
    
    p_val_handle = Line2D([0], [0], color='grey', linestyle='--', label='p-value = 0.05')
    trendline_handle = Line2D([0], [0], color='grey', linestyle=':', label='Trendline (Lowess)')
    
    handles.extend([p_val_handle, trendline_handle])
    
    labels = [h.get_label() for h in handles]
    labels = [str(l).replace('_', ' ') for l in labels]
    
    order = ['Full Model', 'Clinical Sleep Model', 'Clinical Only Model', 'p-value = 0.05', 'Trendline (Lowess)']
    
    label_map = dict(zip(labels, handles))
    ordered_handles = [label_map[lbl] for lbl in order if lbl in label_map]
    ordered_labels = [lbl for lbl in order if lbl in label_map]
    
    ax.legend(handles=ordered_handles, labels=ordered_labels)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as '{filename}'")
    
def plot_reliable_features(mi_df, filename):
    print("Generating Reliable Features Bar Plot...")
    if mi_df.empty: print("Skipping Reliable Features Plot due to missing MI data."); return
    
    df_filtered = mi_df[(mi_df['mi_score'] > 0.2) & (mi_df['p_value'] < 0.05)].copy()
    
    if df_filtered.empty: print("No reliable features found. Skipping plot."); return
    
    feature_order = df_filtered.groupby('feature')['mi_score'].max().sort_values(ascending=False).index
    df_filtered['Model'] = df_filtered['Model'].str.replace('_Model', '').str.replace('_', ' ')

    plt.figure(figsize=(10, len(feature_order) * 0.8 + 2))
    sns.barplot(data=df_filtered, y='feature', x='mi_score', hue='Model', order=feature_order, palette=PALETTE, edgecolor='black', alpha=0.8)
    plt.title('Most Reliable Features (MI > 0.2 & p < 0.05)', fontsize=16); plt.xlabel('MI Score'); plt.ylabel('Feature')
    plt.tight_layout(); plt.savefig(filename, dpi=300); plt.close(); print(f"Plot saved as '{filename}'")


def plot_complete_comparison(performance_df, filename="Complete_Performance_Comparison.png"):

    if performance_df.empty:
        print("No data available to plot.")
        return

    print(f"Generating complete performance comparison plot...")

    hue_order = ['Full', 'Clinical Sleep', 'Clinical Only']
    
    g = sns.catplot(
        data=performance_df,
        x='Scenario', 
        y='Value', 
        hue='Model',
        row='Metric', 
        col='Target',
        kind='bar',
        palette=PALETTE,
        height=5, 
        aspect=1.2,
        legend=True,
        sharey=False,
        order=['All Features', 'MI Top 20', 'Significant MI Features'],
        hue_order=hue_order,
        edgecolor='black'
    )

    g.fig.suptitle('Comprehensive Model Performance Comparison', fontsize=20, y=1.03)

    g.map_dataframe(sns.stripplot, x='Scenario', y='Value', hue='Model', 
                    dodge=True, palette=PALETTE, edgecolor='black', 
                    linewidth=0.5, alpha=0.8, legend=False,
                    order=['All Features', 'MI Top 20', 'Significant MI Features'],
                    hue_order=['Full', 'Clinical Sleep', 'Clinical Only'])

    g.set_axis_labels("Feature Selection Scenario", "Metric Value")
    g.set_titles("Target: {col_name} | Metric: {row_name}")
    g.set_xticklabels(["All Features", "MI Top 20", "Significant MI Features"], rotation=30, ha='right')

    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as '{filename}'")


if __name__ == '__main__':
    MODEL_NAMES = ['Full_Model', 'Clinical_Sleep_Model', 'Clinical_Only_Model']
    SCENARIOS = ['All Features', 'MI Top 20', 'Significant MI Features']

    p_values_df = load_p_values_from_json('p_values.json')

    performance_data = load_all_performance_data(MODEL_NAMES)

    if not performance_data.empty and not p_values_df.empty:
        for scenario in SCENARIOS:
            safe_scenario_name = scenario.replace(' ', '_').replace('<', 'lt')
            filename = f"Model_Performance_Comparison_{safe_scenario_name}.png"
            plot_performance_comparison(performance_data, p_values_df, scenario, filename)
    else:
        print("Could not generate plots due to missing performance or p-value data.")

    print("\nAll visualization tasks are complete.")
    
    mi_data = load_all_mi_data(MODEL_NAMES)
    plot_volcano(mi_data, filename="Feature_Volcano_Plot.png")
    plot_reliable_features(mi_data, filename="Reliable_Features_Bar_Plot.png")
    plot_complete_comparison(performance_data, filename="Overall_Model_Performance.png")
    print("\nAll visualization tasks are complete.")