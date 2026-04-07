import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. データの読み込み
df = pd.read_csv('MONDO_BASE_match.csv')

# 2. 設定
tools = ['PubCaseFinder', 'PhenotypeSearch', 'ZeroShot', 'GestaltMatcher', 'FinalDiagnosis']
ranks = [1, 2, 3, 4, 5]
total_cases = len(df)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 3. 集計処理
plot_data = []
for n in ranks:
    for tool in tools:
        m_col = f'{tool}_Match_rank'
        c_col = f'{tool}_Close_rank'
        match_count = len(df[df[m_col] <= n])
        similar_only_count = len(df[(df[c_col] <= n) & ((df[m_col] > n) | df[m_col].isna())])
        plot_data.append({
            'Rank': n, 'Tool': tool,
            'Match': match_count / total_cases,
            'Similar': similar_only_count / total_cases
        })
plot_df = pd.DataFrame(plot_data)

# 4. 描画設定
fig, ax = plt.subplots(figsize=(20, 10))
x_base = np.arange(len(ranks)) 
width = 0.15 

for i, tool in enumerate(tools):
    tool_data = plot_df[plot_df['Tool'] == tool].sort_values('Rank')
    offset = (i - len(tools)/2 + 0.5) * width
    x_pos = x_base + offset
    
    m_vals = tool_data['Match'].values
    s_vals = tool_data['Similar'].values
    
    # 棒グラフ
    ax.bar(x_pos, m_vals, width, color=colors[i], label=tool, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.bar(x_pos, s_vals, width, bottom=m_vals, color=colors[i], alpha=0.4, hatch='//', edgecolor='white', linewidth=0.5)
    
    # %表示とバー直下のツール名ラベル
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
        # バー直下にツール名を配置
        ax.text(x_pos[j], -0.005, tool, ha='right', va='top', rotation=45, fontsize=8)

# 5. 装飾
ax.set_ylim(0, 0.6)
ax.set_xticks(x_base)
ax.set_xticklabels([f'Rank {r}' for r in ranks], fontsize=14, fontweight='bold')
ax.tick_params(axis='x', which='major', pad=60)
ax.legend(loc='upper left', title="Tools", bbox_to_anchor=(1, 1))
ax.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('mondo_match_plot_v5.png')