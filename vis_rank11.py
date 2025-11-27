import pandas as pd
import os

def analyze_and_report_low_rank_cases():
    """
    GPT-4oのランキングで11位以下になったケースを抽出し、
    分析結果を見やすいMarkdownファイルに出力する。
    """
    # --- ファイルパスの設定 ---
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, "evaluation_summary_with_similarity.csv")
        output_report_path = os.path.join(base_dir, "low_rank_analysis_report.md")
        
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {csv_path}")
        return

    # --- データの抽出とフィルタリング ---
    df['rank_numeric'] = pd.to_numeric(df['GPT4o_Ranked_rank'], errors='coerce')
    is_hit = df['GPT4o_Ranked_res'].isin(['Match', 'Similar'])
    is_low_rank = df['rank_numeric'] >= 11
    low_rank_cases = df[is_hit & is_low_rank].copy()

    # --- レポートコンテンツの生成 ---
    report_content = "# GPT-4o Low Rank Analysis Report\n\n"

    if low_rank_cases.empty:
        report_content += "GPT-4oのランキングで11位以上になった正解ケースはありませんでした。\n"
    else:
        report_content += "## Summary\n"
        report_content += "以下のケースでは、GPT-4oの最終統合ランキングで正解疾患の順位が**11位以下**と低くなっています。\n"
        report_content += "各ツールでの順位を比較することで、ランキングが低下した原因の分析に役立てます。\n\n"
        
        # 表示するカラムを選択
        columns_to_display = [
            'patient_id',
            'correct_omim_id',
            'GestaltMatcher_rank',
            'PCFNode_rank',
            'DiseaseSearch_rank',
            'ZeroShot_rank',
            'GPT4o_Ranked_rank'
        ]
        
        # ランクを整数で表示するためにNaNを-1などに置換してから型変換
        for col in columns_to_display:
            if 'rank' in col:
                low_rank_cases[col] = pd.to_numeric(low_rank_cases[col], errors='coerce').fillna(-1).astype(int)
                low_rank_cases[col] = low_rank_cases[col].apply(lambda x: 'N/A' if x == -1 else x)

        # DataFrameをMarkdownテーブルに変換
        report_content += low_rank_cases[columns_to_display].to_markdown(index=False)

    # --- ファイルへの書き込み ---
    try:
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print(f"分析レポートをMarkdownファイルとして保存しました: {output_report_path}")
    except Exception as e:
        print(f"レポートファイルの書き込み中にエラーが発生しました: {e}")


if __name__ == "__main__":
    analyze_and_report_low_rank_cases()