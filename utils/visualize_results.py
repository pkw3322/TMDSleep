import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches



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
    