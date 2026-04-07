import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

# 保存先
output_dir = './validationCode'
os.makedirs(output_dir, exist_ok=True)

# 1. データの読み込み
with open('./validationCode/mondo-base.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

graph = data['graphs'][0]
edges = [e for e in graph.get('edges', []) if e.get('pred') == 'is_a']

# 2. 統計データの作成
parent_child_counts = Counter(edge['obj'] for edge in edges)
all_subs = {edge['sub'] for edge in edges}
all_objs = set(parent_child_counts.keys())
leaves = all_subs - all_objs
pol_ids = {edge['obj'] for edge in edges if edge['sub'] in leaves}

datasets = [
    {"name": "All Non-Leaf Nodes", "counts": np.array(list(parent_child_counts.values()))},
    {"name": "Parents of Leaves Only", "counts": np.array([parent_child_counts[node_id] for node_id in pol_ids])}
]

# 3. カテゴリビンの定義
BINS = [1, 2, 3, 4, 5, 6, 11, 21, 51, 101, float('inf')]
LABELS = ['1', '2', '3', '4', '5', '6-10', '11-20', '21-50', '51-100', '101+']

def get_categorical_pos(val):
    """数値データがどのラベルのインデックスに対応するかを計算"""
    for i in range(len(BINS)-1):
        if BINS[i] <= val < BINS[i+1]:
            # ビンの中での相対的な位置を線形補間
            width = BINS[i+1] - BINS[i]
            if width == float('inf'): return i
            return i + (val - BINS[i]) / width
    return len(LABELS) - 1

# 4. 可視化
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, ds in zip(axes, datasets):
    counts = ds["counts"]
    p70 = np.percentile(counts, 50) # 実際の70%ラインの値を計算
    
    # カテゴリ集計
    binned_data = [np.sum((counts >= BINS[i]) & (counts < BINS[i+1])) for i in range(len(BINS)-1)]
    
    x_pos = np.arange(len(LABELS))
    bars = ax.bar(x_pos, binned_data, color='lightgreen', edgecolor='black', width=0.8)
    ax.bar_label(bars, padding=3, fontsize=10)
    
    # 赤線の描画（修正ポイント）
    line_pos = get_categorical_pos(p70)
    ax.axvline(line_pos, color='red', linestyle='--', linewidth=2, label=f'50% Line (val={p70:.1f})')

    # グラフ設定
    ax.set_title(f'Distribution: {ds["name"]}', fontsize=14, pad=15)
    ax.set_xlabel('Number of Children (Grouped Bins)', fontsize=12)
    ax.set_ylabel('Frequency (Number of Nodes)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(LABELS)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'combined_distribution_grouped_fixed.png'))