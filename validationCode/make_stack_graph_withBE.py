import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. データの読み込み
df = pd.read_csv('MONDO_BASE_match.csv')

# 2. 設定
# 既存のツール + Best Effort
original_tools = ['PubCaseFinder', 'PhenotypeSearch', 'ZeroShot', 'GestaltMatcher', 'FinalDiagnosis']
# Best Effortの算出対象
best_effort_components = ['PubCaseFinder', 'PhenotypeSearch', 'ZeroShot', 'GestaltMatcher']
all_plot_tools = original_tools + ['Best Effort']

ranks = [1, 2, 3, 4, 5]
total_cases = len(df)
# 色設定（オレンジをBest Effortに割り当て）
colors = ['#1f77b4', "#8B6C50", '#2ca02c', '#d62728', '#9467bd', '#ff7f0e']

# 3. 集計処理
plot_data = []
for n in ranks:
    # 個別ツールの集計
    for tool in original_tools:
        m_col = f'{tool}_Match_rank'
        c_col = f'{tool}_Close_rank'
        match_count = len(df[df[m_col] <= n])
        similar_only_count = len(df[(df[c_col] <= n) & ((df[m_col] > n) | df[m_col].isna())])
        plot_data.append({
            'Rank': n, 'Tool': tool,
            'Match': match_count / total_cases,
            'Similar': similar_only_count / total_cases
        })
    
    # Best Effortの集計
    # 指定された4つのツールのいずれかでMatchがあればMatch、SimilarがあればSimilar
    match_mask = pd.Series([False] * len(df))
    close_mask = pd.Series([False] * len(df))
    for tool in best_effort_components:
        match_mask |= (df[f'{tool}_Match_rank'] <= n)
        close_mask |= (df[f'{tool}_Close_rank'] <= n)
    
    be_match_count = len(df[match_mask])
    be_similar_only_count = len(df[close_mask & ~match_mask])
    
    plot_data.append({
        'Rank': n, 'Tool': 'Best Effort',
        'Match': be_match_count / total_cases,
        'Similar': be_similar_only_count / total_cases
    })

plot_df = pd.DataFrame(plot_data)

# 4. 描画設定
fig, ax = plt.subplots(figsize=(20, 10))
x_base = np.arange(len(ranks)) 
width = 0.12 # 6本のバーを表示するために調整

for i, tool in enumerate(all_plot_tools):
    tool_data = plot_df[plot_df['Tool'] == tool].sort_values('Rank')
    offset = (i - len(all_plot_tools)/2 + 0.5) * width
    x_pos = x_base + offset
    
    m_vals = tool_data['Match'].values
    s_vals = tool_data['Similar'].values
    
    # 棒グラフ
    ax.bar(x_pos, m_vals, width, color=colors[i], label=tool, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.bar(x_pos, s_vals, width, bottom=m_vals, color=colors[i], alpha=0.4, hatch='//', edgecolor='white', linewidth=0.5)
    
    # %表示ラベル
    for j, (m, s) in enumerate(zip(m_vals, s_vals)):
        # Match %
        if m > 0.015:
            ax.text(x_pos[j], m/2, f'{m*100:.1f}%', ha='center', va='center', color='white', fontsize=7, fontweight='bold', rotation=90)
        # Similar %
        if s > 0.01:
            ax.text(x_pos[j], m + s/2, f'{s*100:.1f}%', ha='center', va='center', color='black', fontsize=7, rotation=90)
        # 合計 (Total %)
        if (m + s) > 0:
            ax.text(x_pos[j], m + s + 0.005, f'{(m+s)*100:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x_base)
ax.set_xticklabels([f'Rank {n}' for n in ranks])
ax.set_ylabel('Percentage of cases')
ax.set_title('Diagnostic Performance with Best Effort (Top-N Analysis)')
ax.set_ylim(0, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend()

plt.tight_layout()
plt.savefig('MONDO_match_plot_best_effort.png')