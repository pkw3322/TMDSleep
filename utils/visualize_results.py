import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_performance_with_error_bars(results_a, results_b, results_c, metric_name, target):
    """Visualize model performance with error bars."""
    
    metric_mean_key = metric_name.lower()
    metric_std_key = metric_name.lower() + '_std'
    
    plot_data = []
    model_names = {
        'A': 'Full Model (Base Features)',
        'B': 'Clinical + Sleep Model (without Patient Report)',
        'C': 'Clinical-Only Model (without Patient Report & Sleep Metrics)'
    }
    all_results = {'A': results_a, 'B': results_b, 'C': results_c}
    plot_data = []
 
    for model_key, scenarios in all_results.items():
        for scenario, targets in scenarios.items():
            if target in targets:
                metrics = targets[target]
                plot_data.append({
                    'Scenario': scenario,
                    'Mean': metrics[metric_mean_key],
                    'Std': metrics[metric_std_key],
                    'Model': model_names[model_key]
                })

    if not plot_data:
        print(f"No data to plot for {target} - {metric_name}")
        return
        
    df_plot = pd.DataFrame(plot_data)
    
    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#000080', '#EA4335', '#34A853']
    light_colors = ['#8c8cd9', '#f5a49b', '#97d1a9']
    
    scenarios = df_plot['Scenario'].unique()
    models = [model_names['A'], model_names['B'], model_names['C']]
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, model in enumerate(models):
        model_data = df_plot[df_plot['Model'] == model].set_index('Scenario').reindex(scenarios).reset_index()
        means = model_data['Mean']
        stds = model_data['Std']
        pos = x - width + (i * width)
        
        ax.bar(pos, means + stds, width, color=light_colors[i], edgecolor='grey', alpha=0.7)
        ax.bar(pos, means, width, color=colors[i], edgecolor='black')

    y_max = (df_plot['Mean'] + df_plot['Std']).max()
    ax.set_ylim(0, y_max * 1.15)

    ax.set_title(f'Model Performance for {target} Prediction ({metric_name.upper()})', fontsize=20, pad=20)
    ax.set_xlabel('Feature Selection Scenario', fontsize=14)
    ax.set_ylabel(f'{metric_name.upper()} Value', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)

    handles = [mpatches.Patch(color=colors[i], label=model) for i, model in enumerate(models)]
    handles.append(mpatches.Patch(color='grey', alpha=0.4, label='Mean + Std (Lighter Shade)'))
    
    ax.legend(handles=handles, title='Model', fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'Performance_{target}_{metric_name.upper()}.png', dpi=300)
    print(f'Figure saved: Performance_{target}_{metric_name.upper()}.png')
    plt.show()


def plot_combined_metric_results(results_a, results_b, results_c, metric_name):
    """Visualizes the performance metrics of three models in a single grouped bar chart."""
    plot_data = []

    model_names = {
        'A': 'Full Model (Base Features)',
        'B': 'Clinical + Sleep Model (without Patient Report)',
        'C': 'Clinical-Only Model (without Patient Report & Sleep Metrics)'
    }
    
    
    all_results = {'A': results_a, 'B': results_b, 'C': results_c}

    for model_key, results in all_results.items():
        for scenario, res_data in results.items():
            if not res_data: continue
            for target, metrics in res_data.items():
                if metric_name.lower() in metrics:
                    plot_data.append({
                        'Scenario': scenario,
                        'Target': target,
                        'Metric': metrics[metric_name.lower()],
                        'Model': model_names[model_key]
                    })
    
    if not plot_data:
        print(f"No data to plot for metric: {metric_name}")
        return
        
    df_plot = pd.DataFrame(plot_data)
    
    # CMI Plot
    plt.figure(figsize=(16, 10))
    ax_cmi = sns.barplot(x='Scenario', y='Metric', hue='Model', data=df_plot[df_plot['Target'] == 'CMI'], palette=['#000080', '#EA4335', '#34A853'])
    plt.title(f'CMI Model Performance Comparison ({metric_name.upper()})', fontsize=18, pad=20)
    plt.ylabel(f'{metric_name.upper()} Value', fontsize=14)
    plt.xlabel('Feature Selection Scenario', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.legend(title='Model', fontsize=12, title_fontsize=13)

    for container in ax_cmi.containers:
        ax_cmi.bar_label(container, fmt='%.4f', fontsize=10, padding=3)
        
    plt.tight_layout(pad=3.0)
    plt.savefig(f'Model_Comparison_{metric_name.upper()}_CMI.png', dpi=300)
    plt.show()
    plt.close()
    
    # VAS Plot
    plt.figure(figsize=(16, 10))
    ax_vas = sns.barplot(x='Scenario', y='Metric', hue='Model', data=df_plot[df_plot['Target'] == 'VAS'], palette=['#000080', '#EA4335', '#34A853'])
    plt.title(f'VAS Model Performance Comparison ({metric_name.upper()})', fontsize=18, pad=20)
    plt.ylabel(f'{metric_name.upper()} Value', fontsize=14)
    plt.xlabel('Feature Selection Scenario', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Model', fontsize=12, title_fontsize=13)

    for container in ax_vas.containers:
        ax_vas.bar_label(container, fmt='%.4f', fontsize=10, padding=3)

    plt.tight_layout(pad=3.0)
    plt.savefig(f'Model_Comparison_{metric_name.upper()}_VAS.png', dpi=300)
    plt.show()
    plt.close()
    print(f"Combined {metric_name.upper()} comparison plots saved successfully.")
    
def plot_importance_comparison(scores_a, scores_b, scores_c, top_n=30):
    
    model_names = {'A': 'Full Model', 'B': 'Clinical + Sleep Model', 'C': 'Clinical-Only Model'}
    df_a = pd.DataFrame(scores_a, columns=['Variable', 'Importance', 'p_value']); df_a['Model'] = model_names['A']
    df_b = pd.DataFrame(scores_b, columns=['Variable', 'Importance', 'p_value']); df_b['Model'] = model_names['B']
    df_c = pd.DataFrame(scores_c, columns=['Variable', 'Importance', 'p_value']); df_c['Model'] = model_names['C']
    
    combined_df = pd.concat([df_a, df_b, df_c])
    
    top_vars = df_a.nlargest(top_n, 'Importance')['Variable']
    plot_df = combined_df[combined_df['Variable'].isin(top_vars)]
    
    sig_vars = set(plot_df[plot_df['p_value'] < 0.05]['Variable'])
    y_labels = [f"{var} *" if var in sig_vars else var for var in top_vars]

    plt.style.use('seaborn-v0_8-talk')
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(ax=ax, y='Variable', x='Importance', hue='Model', data=plot_df,
                palette=['#000080', '#EA4335', '#34A853'], order=top_vars, orient='h')
    
    ax.set_yticklabels(y_labels)
    ax.set_title('Comparison of Feature Importance by Model (MI Score)', fontsize=20, pad=20)
    ax.set_xlabel('Mutual Information (MI) Score', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='white', label='* p < 0.05 in at least one model'))
    ax.legend(handles=handles, title='Model', fontsize=12)

    plt.tight_layout()
    plt.savefig('Feature_Importance_Comparison_with_Sig.png', dpi=300)
    print('Figure saved: Feature_Importance_Comparison_with_Sig.png')
    plt.show()
    

def visulaize_scenario(final_results_cmi, final_results_vas):
    # --- 시각화를 위한 데이터 정제 (Data Wrangling) ---
    # 분석 결과를 시각화에 용이한 긴 형태의 데이터프레임으로 변환하는 함수
    def process_results_for_plotting(results_dict, target_name):
        plot_data = []
        for scenario, models in results_dict.items():
            for model, metrics in models.items():
                for metric, scores in metrics.items():
                    for score in scores:
                        plot_data.append([target_name, scenario, model, metric, score])
        return pd.DataFrame(plot_data, columns=['Target', 'Scenario', 'Model', 'Metric', 'Value'])

                      
    df_cmi = process_results_for_plotting(final_results_cmi, 'CMI')
    df_vas = process_results_for_plotting(final_results_vas, 'VAS')
    plot_df = pd.concat([df_cmi, df_vas]) # 실제 모든 메트릭이 포함된 데이터프레임이라 가정


    # --- 시각화 실행 ---
    # catplot을 사용하여 Metric(행)과 Target(열)에 따라 그래프를 나눕니다.
    g = sns.catplot(
        data=plot_df,  # << 실제 결과가 담긴 데이터프레임 사용
        x='Model',
        y='Value',
        hue='Model',
        row='Metric',
        col='Target',
        kind='box',
        palette='Set2',
        sharey=False,
        legend=False,
        boxprops=dict(alpha=0.6)
    )

    # 각 박스플롯 위에 jitter plot(stripplot)을 겹쳐 그립니다.
    g.map_dataframe(
        sns.stripplot,
        x='Model',
        y='Value',
        hue='Model',
        dodge=True,
        edgecolor='gray',
        linewidth=0.5,
        palette='Set2',
        alpha=0.8,
        jitter=0.2
    )

    # 제목 및 레이블 설정
    g.fig.suptitle('Model Performance with Actual Cross-Validation Scores', y=1.03, fontsize=16)
    g.set_axis_labels("Model", "Metric Value")
    g.set_titles("Target: {col_name} | Metric: {row_name}")
    g.set_xticklabels(rotation=15)

    # 범례 처리 (중복 제거 후 하나만 표시)
    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    g.fig.legend(handles[:len(plot_df['Model'].unique())], labels[:len(plot_df['Model'].unique())], 
                title='Model', loc='upper right', bbox_to_anchor=(1, 1))

    # 각 subplot의 범례는 숨김
    for ax in g.axes.flat:
        ax.get_legend().remove() if ax.get_legend() else None

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("model_performance_boxplot_with_jitter.png", dpi=300)
    plt.show()


def load_and_process_mi_scores(mi_score_files):
    """Reads MI Score JSON files for multiple models and combines them into a single DataFrame."""
    all_mi_data = []
    for model_name, file_path in mi_score_files.items():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df['model'] = model_name
            all_mi_data.append(df)
        except FileNotFoundError:
            print(f"Warning: Could not find file '{file_path}'. Skipping.")
    
    if not all_mi_data:
        return pd.DataFrame()

    mi_df = pd.concat(all_mi_data, ignore_index=True)
    mi_df['-log10(p_value)'] = -np.log10(mi_df['p_value'] + 1e-10)
    bins = [-np.inf, 0.05, 0.2, 0.5, np.inf]
    labels = ['Negligible', 'Weakly informative', 'Moderately informative', 'Highly informative']
    mi_df['informativeness'] = pd.cut(mi_df['mi_score'], bins=bins, labels=labels)
    return mi_df

def visualize_analysis_summary(mi_score_files, raw_data, alpha=0.05):
    """Generates a comprehensive visualization of MI Score analysis and CMI/VAS correlation, including marginal histograms."""
    mi_df = load_and_process_mi_scores(mi_score_files)
    if mi_df.empty:
        print("Error: Could not load MI Score data.")
        return

    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    
    # fig.suptitle('Comprehensive Analysis of Feature Importance and Target Correlation', fontsize=22)

    sns.scatterplot(
        data=mi_df, x='mi_score', y='-log10(p_value)', hue='informativeness',
        style='model', s=100, alpha=0.8, ax=ax1, palette='viridis'
    )
    ax1.axhline(-np.log10(alpha), color='r', linestyle='--', label=f'p-value = {alpha}')
    ax1.set_title('Feature Significance and Informativeness (Volcano Plot)', fontsize=16, pad=15)
    ax1.legend(title='Informativeness')

    reliable_features = mi_df[(mi_df['mi_score'] > 0.2) & (mi_df['p_value'] < alpha)]
    if not reliable_features.empty:
        sns.barplot(data=reliable_features.sort_values('mi_score', ascending=False),
                    x='mi_score', y='feature', hue='model', ax=ax2, palette='Set2')
        ax2.set_title('Most Reliable Features (MI > 0.2 & p < 0.05)', fontsize=16, pad=15)
    else:
        ax2.text(0.5, 0.5, 'No reliable features found.', ha='center', va='center')
        ax2.set_title('Most Reliable Features', fontsize=16, pad=15)

    if 'CMI' in raw_data.columns and 'VAS' in raw_data.columns:
        divider = make_axes_locatable(ax3)
        ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax3)
        ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax3)
        
        ax_histx.set_title('Correlation between CMI and VAS', fontsize=16, pad=15)

        sns.regplot(
            x='CMI', y='VAS', data=raw_data, ax=ax3,
            scatter_kws={'alpha': 0.4}, line_kws={'color': 'red'}
        )
        corr, pval = pearsonr(raw_data['CMI'].dropna(), raw_data['VAS'].dropna())
        annotation_text = f'Pearson r = {corr:.3f}\np-value = {pval:.3f}'
        ax3.text(0.95, 0.95, annotation_text, transform=ax3.transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax3.grid(True, linestyle='--' , alpha=0.5)

        sns.histplot(data=raw_data, x='CMI', ax=ax_histx, bins=20)
        sns.histplot(data=raw_data, y='VAS', ax=ax_histy, bins=20)

        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

    else:
        ax3.text(0.5, 0.5, "'CMI' or 'VAS' columns not found.", ha='center', va='center')
        ax3.set_title('Correlation between CMI and VAS', fontsize=16, pad=15)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig("analysis_summary_with_jointplot.png", dpi=300)
    plt.show()

def visualize_cmi_vas_correlation(raw_data):
    
    if 'CMI' not in raw_data.columns or 'VAS' not in raw_data.columns:
        print("Error: 'raw_data' DataFrame must contain both 'CMI' and 'VAS' columns.")
        return

    print("Generating CMI vs. VAS correlation plot...")

    # Calculate Pearson correlation
    corr, pval = pearsonr(raw_data['CMI'].dropna(), raw_data['VAS'].dropna())
    
    # Create the jointplot
    g = sns.jointplot(
        data=raw_data,
        x='CMI',
        y='VAS',
        kind='reg', # Adds a regression line
        height=8,
        scatter_kws={'alpha': 0.4, 's': 50},
        line_kws={'color': 'red', 'lw': 2}
    )
    
    # Add annotation text for correlation
    annotation_text = f'Pearson r = {corr:.3f}\np-value = {pval:.3f}'
    g.ax_joint.text(
        0.05, 0.95, annotation_text,
        transform=g.ax_joint.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5)
    )
    
    g.fig.suptitle('Correlation between CMI and VAS with Marginal Distributions', y=1.02, fontsize=16)
    
    plt.savefig("CMI_VAS_Correlation.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot saved as CMI_VAS_Correlation.png")
    
    

def process_results_for_plotting(results_dict, target_name):

    plot_data = []
    for scenario, models in results_dict.items():
        for model, metrics in models.items():
            for metric, scores in metrics.items():
                for score in scores:
                    plot_data.append([target_name, scenario, model, metric, score])
    
    df = pd.DataFrame(plot_data, columns=['Target', 'Scenario', 'Model', 'Metric', 'Value'])
    return df

def visualize_results(cmi_json_path, vas_json_path):

    try:
        with open(cmi_json_path, 'r') as f:
            cmi_results = json.load(f)
        with open(vas_json_path, 'r') as f:
            vas_results = json.load(f)
    except FileNotFoundError as e:
        return
    except json.JSONDecodeError as e:
        return

    df_cmi = process_results_for_plotting(cmi_results, 'CMI')
    df_vas = process_results_for_plotting(vas_results, 'VAS')
    
    plot_df = pd.concat([df_cmi, df_vas], ignore_index=True)
    

    g = sns.catplot(
        data=plot_df,
        x='Model',
        y='Value',
        hue='Model',
        row='Metric',
        col='Target',
        kind='box',
        palette='Set2',
        sharey=False,
        legend=False,
        boxprops=dict(alpha=0.7)
    )
    
    g.map_dataframe(
        sns.stripplot,
        x='Model',
        y='Value',
        hue='Model',
        dodge=True,
        edgecolor='gray',
        linewidth=0.5,
        palette='Set2',
        alpha=0.8
    )

    g.fig.suptitle('Model Performance Comparison based on Cross-Validation Scores', y=1.03, fontsize=16)
    g.set_axis_labels("", "Metric Value")
    g.set_titles("Target: {col_name} | Metric: {row_name}")
    
    g.set_xticklabels(['Full', 'Clinical+Sleep', 'Clinical-Only'], rotation=15, ha='right')
    
    labels = plot_df['Model'].unique().tolist()

    palette = sns.color_palette('Set2', n_colors=len(labels))
    handles = [plt.Rectangle((0,0),1,1, color=palette[i]) for i in range(len(labels))]

   
    new_labels = [f"{label}" for label in labels]

    for ax in g.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
         
    plt.tight_layout(rect=[0, 0.05, 1, 0.96]) # 하단 여백 추가
    g.fig.set_size_inches(12, 10)
    plt.savefig("Model_Performance_Comparison.png", dpi=300, bbox_inches='tight')
    
    plt.show()
