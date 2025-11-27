import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def plot_tool_performance():
    """
    evaluation_summary_with_similarity.csvを読み込み、
    各ツールの正解率を棒グラフで表示する。
    """
    # --- ファイルパスの設定 ---
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "evaluation_summary_with_similarity.csv")
        output_image_path = os.path.join(base_dir, "tool_performance_summary.png")
        
        # CSVファイルを読み込む
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {csv_path}")
        print("evaluate_ranks_advanced.py を先に実行して、評価ファイルを生成してください。")
        return

    # --- データの集計 ---
    # 解析対象のツールの結果カラム
    tool_res_columns = {
        "GestaltMatcher_res": "GestaltMatcher",
        "PCFNode_res": "PubCaseFinder",
        "DiseaseSearch_res": "Pheno. Search",
        "ZeroShot_res": "Zero-Shot",
        "GPT4o_Ranked_res": "GPT-4o Ranked"
    }
    
    total_cases = len(df)
    performance_data = {}

    for col, name in tool_res_columns.items():
        # "Match" または "Similar" の数をカウント
        hits = df[col].isin(["Match", "Similar"]).sum()
        # 正解率（ヒット率）を計算
        hit_rate = (hits / total_cases) * 100 if total_cases > 0 else 0
        performance_data[name] = hit_rate

    # 集計結果をPandas DataFrameに変換
    performance_df = pd.DataFrame(list(performance_data.items()), columns=['Tool', 'Hit Rate (%)'])
    performance_df = performance_df.sort_values(by='Hit Rate (%)', ascending=False)

    # --- グラフの描画 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # 棒グラフを作成
    bars = sns.barplot(x='Tool', y='Hit Rate (%)', data=performance_df, ax=ax, palette='viridis')

    # 各棒の上にパーセンテージを表示
    for bar in bars.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5, # 少し上に表示
            f'{bar.get_height():.1f}%', # 小数点以下1桁まで
            ha='center',
            va='bottom',
            fontsize=11,
            color='black'
        )

    # グラフの装飾
    ax.set_title('Diagnostic Tool Performance (Hit Rate)', fontsize=16)
    ax.set_xlabel('Tool / Method', fontsize=12)
    ax.set_ylabel('Hit Rate (%) [Match or Similar]', fontsize=12)
    ax.set_ylim(0, 100) # Y軸の範囲を0-100%に設定
    plt.xticks(rotation=0) # X軸ラベルの回転をなくす

    # レイアウトを調整して画像を保存
    plt.tight_layout()
    plt.savefig(output_image_path)

    print(f"グラフを {output_image_path} に保存しました。")
    
    # グラフを表示
    plt.show()


if __name__ == "__main__":
    plot_tool_performance()