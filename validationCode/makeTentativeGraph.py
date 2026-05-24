import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_tentative_diagnosis_plot(csv_path, output_image):
    # 1. データの読み込み
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} が見つかりません。")
        return

    # 2. 対象ツールの設定
    tool = 'TentativeDiagnosis'
    m_col = f'{tool}_Match_rank'
    c_col = f'{tool}_Close_rank'

    # 3. データ内に存在する全ての順位（Rank）を特定
    # Match_rankとClose_rankの両方からユニークな数値を取得
    all_ranks = pd.concat([df[m_col], df[c_col]]).dropna().unique()
    if len(all_ranks) == 0:
        print("表示できるランクデータがありません。")
        return
    
    ranks = sorted([int(r) for r in all_ranks])
    total_cases = len(df)
    plot_data = []

    # 4. 各Rankにおける累積的中率を計算
    for n in ranks:
        # Match: 指定順位までに完全一致（またはMONDO同一）がある
        match_count = len(df[df[m_col] <= n])
        # Similar: Matchではないが、指定順位までにClose Matchがある
        similar_only_count = len(df[(df[c_col] <= n) & ((df[m_col] > n) | df[m_col].isna())])
        
        plot_data.append({
            'Rank': n,
            'Match': match_count / total_cases,
            'Similar': similar_only_count / total_cases
        })

    plot_df = pd.DataFrame(plot_data)

    # 5. グラフの描画設定
    # ランク数に応じて横幅を自動調整
    fig, ax = plt.subplots(figsize=(max(12, len(ranks)*0.8), 8))
    x_pos = np.arange(len(ranks))
    width = 0.7
    color = '#9467bd' # TentativeDiagnosis用の紫系カラー

    # スタックバー（累積）の描画
    m_vals = plot_df['Match'].values
    s_vals = plot_df['Similar'].values

    # 下段：Match
    ax.bar(x_pos, m_vals, width, color=color, label='Match', alpha=0.9, edgecolor='white')
    # 上段：Similar (Matchの上に積む)
    ax.bar(x_pos, s_vals, width, bottom=m_vals, color=color, label='Similar (Close Match)', 
           alpha=0.4, hatch='//', edgecolor='white')

    # 6. 数値ラベルの追加（バーの内部と頂上）
    for i, (m, s) in enumerate(zip(m_vals, s_vals)):
        # Match部分のラベル（白文字、縦書き）
        if m > 0.03:
            ax.text(x_pos[i], m/2, f'{m*100:.1f}%', ha='center', va='center', 
                    color='white', fontsize=8, fontweight='bold', rotation=90)
        
        # Similar部分のラベル（黒文字、縦書き）
        if s > 0.03:
            ax.text(x_pos[i], m + s/2, f'{s*100:.1f}%', ha='center', va='center', 
                    color='black', fontsize=8, rotation=90)
        
        # バーの頂上に合計値を表示
        total = m + s
        if total > 0:
            ax.text(x_pos[i], total + 0.005, f'{total*100:.1f}%', ha='center', 
                    va='bottom', fontsize=9, fontweight='bold')

    # 7. 軸とレイアウトの調整
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Rank {n}' for n in ranks])
    ax.set_ylabel('Percentage of cases (N={})'.format(total_cases))
    ax.set_title(f'Diagnostic Performance: {tool} (All Ranks)', fontsize=14)
    ax.set_ylim(0, 1.1) # ラベル表示用に少し余裕を持たせる
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_image)
    print(f"Graph saved as: {output_image}")

if __name__ == "__main__":
    create_tentative_diagnosis_plot('MONDO_BASE_match.csv', 'tentative_diagnosis_full_ranks_labeled.png')