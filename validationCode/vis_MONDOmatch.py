import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. データの読み込み
df = pd.read_csv('MONDO_BASE_match.csv')
tools = ['PubCaseFinder', 'PhenotypeSearch', 'ZeroShot', 'GestaltMatcher', 'FinalDiagnosis']

# 2. 結果の内訳（Match/Similar/Miss）の集計と%計算
res_data = []
total_all = len(df)
for tool in tools:
    counts = df[f'{tool}_res'].value_counts()
    res_data.append({
        'Tool': tool,
        'Match': counts.get('Match', 0),
        'Similar': counts.get('Similar', 0),
        'Miss': counts.get('Miss', 0)
    })
df_res = pd.DataFrame(res_data).set_index('Tool')

# 3. ランク分布（1~5）の集計と%計算
rank_counts = pd.DataFrame(index=tools, columns=[1, 2, 3, 4, 5]).fillna(0)
rank_annots = pd.DataFrame(index=tools, columns=[1, 2, 3, 4, 5]).fillna("")

for tool in tools:
    # Match または Similar のケースのみ抽出
    mask = df[f'{tool}_res'].isin(['Match', 'Similar'])
    success_df = df[mask]
    total_success = len(success_df)
    
    # ランク1~5の個数をカウント
    counts = pd.to_numeric(success_df[f'{tool}_rank'], errors='coerce').value_counts()
    for r in range(1, 6):
        c = int(counts.get(float(r), 0))
        rank_counts.loc[tool, r] = c
        # 成功数の中での割合を計算
        if total_success > 0:
            pct = (c / total_success) * 100
            rank_annots.loc[tool, r] = f"{c}\n({pct:.1f}%)"

# --- 4. グラフの描画 ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 左: 積層棒グラフ（全体に占める割合）
colors = ['#4CAF50', '#FFC107', '#F44336']
ax0 = df_res[['Match', 'Similar', 'Miss']].plot(kind='bar', stacked=True, color=colors, ax=axes[0])

# 棒グラフ内に件数と%を表示
for container in ax0.containers:
    labels = []
    for bar in container:
        val = bar.get_height()
        if val > 0:
            pct = (val / total_all) * 100
            labels.append(f"{int(val)}\n({pct:.1f}%)")
        else:
            labels.append("")
    ax0.bar_label(container, labels=labels, label_type='center', fontsize=9)

axes[0].set_title('Overall Prediction Outcomes (Match/Similar/Miss %)', fontsize=14)
axes[0].set_ylabel('Number of Cases')
axes[0].legend(title='Result', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0].tick_params(axis='x', rotation=45)

# 右: ランク分布のヒートマップ（成功数の中での割合）
sns.heatmap(rank_counts.astype(int), annot=rank_annots.values, fmt="", cmap="YlGnBu", 
            cbar_kws={'label': 'Count'}, ax=axes[1])
axes[1].set_title('Rank Distribution (1-5) for Successful Matches\n(Count and % within successful cases)', fontsize=14)
axes[1].set_xlabel('Rank')
axes[1].set_ylabel('Tool')

plt.tight_layout()
plt.savefig('evaluation_summary_with_pct.png')
plt.show()