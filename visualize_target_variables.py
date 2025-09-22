import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec
from data.data_preprocessing import preprocessing_data


if __name__ == "__main__":
    df, _ = preprocessing_data()


    
    print("Generating and saving Figure (A)...")
    plt.figure(figsize=(7, 7))
    ax1 = plt.gca() # Get Current Axes
    sns.boxplot(y=df['CMI'], ax=ax1, color='#bee3f8', width=0.3, fliersize=0)
    sns.stripplot(y=df['CMI'], ax=ax1, color='#8ecae6', alpha=0.4, jitter=0.25)
    ax1.set_title('(A) Distribution of CMI', fontsize=14, weight='bold')
    ax1.set_ylabel('CMI Score')
    ax1.set_xlabel('')
    ax1.set_xticks([])
    plt.tight_layout()
    plt.savefig('figure_A_CMI_distribution.png', dpi=300)
    plt.close()
    print("Figure (A) saved as 'figure_A_CMI_distribution.png'")


    print("\nGenerating and saving Figure (B)...")
    plt.figure(figsize=(7, 7))
    ax2 = plt.gca()
    sns.boxplot(y=df['VAS'], ax=ax2, color='#ffc2d1', width=0.3)
    sns.stripplot(y=df['VAS'], ax=ax2, color='#f582ae', alpha=0.3, jitter=0.25)
    ax2.set_title('(B) Distribution of VAS', fontsize=14, weight='bold')
    ax2.set_ylabel('VAS Score')
    ax2.set_xlabel('')
    ax2.set_xticks([])
    plt.tight_layout()
    plt.savefig('figure_B_VAS_distribution.png', dpi=300)
    plt.close()
    print("Figure (B) saved as 'figure_B_VAS_distribution.png'")


    print("\nGenerating and saving Figure (C) with softer colors...")
    g = sns.jointplot(data=df, x='CMI', y='VAS', kind='reg', height=8,
                  # marginal_kws에 kde=True를 추가하여 분포 곡선을 명시적으로 그림
                  marginal_kws=dict(bins=20, fill=True, kde=True),
                  scatter_kws={'alpha': 0.5, 'color': '#98d8aa', 's': 30, 'edgecolor': 'w'},
                  line_kws={'color': '#d9534f'})

    corr, p_value = pearsonr(df['CMI'], df['VAS'])
    stats_text = f'Pearson r = {corr:.3f}\np-value = {p_value:.3f}'
    bbox_props = dict(boxstyle='round,pad=0.5', fc='white', ec='black', lw=1, alpha=0.8)
    g.ax_joint.text(0.6, 0.05, stats_text, transform=g.ax_joint.transAxes,
                    fontsize=11, verticalalignment='bottom', bbox=bbox_props)

    g.fig.suptitle('(C) Correlation between CMI and VAS', fontsize=14, weight='bold')
    g.fig.tight_layout(rect=(0, 0, 1, 0.96))

    for patch in g.ax_marg_x.patches:
        patch.set_facecolor('#bee3f8')
    if g.ax_marg_x.get_lines():
        g.ax_marg_x.get_lines()[0].set_color('#8ecae6')

    for patch in g.ax_marg_y.patches:
        patch.set_facecolor('#ffc2d1')
    if g.ax_marg_y.get_lines():
        g.ax_marg_y.get_lines()[0].set_color('#f582ae')

    plt.savefig('figure_C_correlation.png', dpi=300)
    plt.close()
    print("Figure (C) saved as 'figure_C_correlation.png'")