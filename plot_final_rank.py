import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

def plot_gpt4o_rank_distribution():
    """
    evaluation_summary_with_similarity.csvを読み込み、
    GPT-4oのランキングで正解が何位に入ったかの分布をプロットする。
    """
    # --- ファイルパスの設定 ---
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "evaluation_summary_with_similarity.csv")
        output_image_path = os.path.join(base_dir, "gpt4o_rank_distribution.png")
        
        # CSVファイルを読み込む
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {csv_path}")
        print("評価ファイルを先に生成してください。")
        return

    # --- データの集計 ---
    # GPT-4oが正解したケース（MatchまたはSimilar）のみを抽出
    correct_cases = df[df['GPT4o_Ranked_res'].isin(['Match', 'Similar'])].copy()

    # 'GPT4o_Ranked_rank' 列を数値型に変換（変換できない値はNaNになる）
    correct_cases['rank_numeric'] = pd.to_numeric(correct_cases['GPT4o_Ranked_rank'], errors='coerce')
    
    # NaN（変換できなかった行）を除外
    correct_cases.dropna(subset=['rank_numeric'], inplace=True)
    
    # 整数型に変換
    correct_cases['rank_numeric'] = correct_cases['rank_numeric'].astype(int)

    # ランクごとの件数をカウント
    rank_counts = correct_cases['rank_numeric'].value_counts().sort_index()

    # --- グラフの描画 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # 棒グラフを作成
    bars = sns.barplot(x=rank_counts.index, y=rank_counts.values, ax=ax, color='skyblue')

    # 各棒の上に件数を表示
    for bar in bars.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f'{int(bar.get_height())}',
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

    # グラフの装飾
    ax.set_title('Distribution of Correct Ranks in GPT-4o Output', fontsize=16)
    ax.set_xlabel('Rank of Correct Diagnosis', fontsize=12)
    ax.set_ylabel('Number of Cases', fontsize=12)
    
    # X軸の目盛りを整数にする
    max_rank = rank_counts.index.max()
    if max_rank > 0:
        ax.set_xticks(range(0, max_rank, 1))
        ax.set_xticklabels(range(1, max_rank + 1, 1))

    plt.xticks(rotation=0)

    # レイアウトを調整して画像を保存
    plt.tight_layout()
    plt.savefig(output_image_path)

    print(f"グラフを {output_image_path} に保存しました。")
    
    # グラフを表示
    plt.show()


if __name__ == "__main__":
    plot_gpt4o_rank_distribution()