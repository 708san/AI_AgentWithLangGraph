import pandas as pd
import os

def filter_summary_by_id_list():
    """
    diagnosis_match_result.csvのIDリストに基づき、
    evaluation_summary_with_similarity.csvをフィルタリングする。
    """
    # --- ファイルパスの設定 ---
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        main_csv_path = os.path.join(base_dir, "evaluation_summary_with_similarity.csv")
        filter_csv_path = os.path.join(base_dir, "escape", "result_analyze", "diagnosis_match_result.csv")
        output_csv_path = os.path.join(base_dir, "evaluation_summary_with_similarity_restricted.csv")

        # --- ファイルの存在チェック ---
        if not os.path.exists(main_csv_path):
            print(f"エラー: メインファイルが見つかりません: {main_csv_path}")
            return
        if not os.path.exists(filter_csv_path):
            print(f"エラー: フィルタ用ファイルが見つかりません: {filter_csv_path}")
            return

        # 1. diagnosis_match_result.csv から許可するIDのリストを取得
        filter_df = pd.read_csv(filter_csv_path)
        # IDを文字列に変換して比較の一貫性を保ち、ユニークな値のセットを作成
        allowed_ids = set(filter_df['patient_id'].astype(str))
        print(f"'{os.path.basename(filter_csv_path)}' から {len(allowed_ids)} 件のユニークな patient_id を読み込みました。")

        # 2. メインの評価ファイルを読み込む
        main_df = pd.read_csv(main_csv_path)
        # こちらのIDも文字列に変換
        main_df['patient_id'] = main_df['patient_id'].astype(str)

        # 3. 許可リストに含まれるIDでフィルタリング
        restricted_df = main_df[main_df['patient_id'].isin(allowed_ids)]

        # 4. 除外されたIDを特定
        all_original_ids = set(main_df['patient_id'])
        excluded_ids = sorted(list(all_original_ids - allowed_ids), key=int)

        # 5. 新しいCSVファイルとして保存
        restricted_df.to_csv(output_csv_path, index=False)
        print(f"\nフィルタリング後のファイルを '{os.path.basename(output_csv_path)}' として保存しました。")
        print(f"（元の {len(main_df)} 件から {len(restricted_df)} 件に絞り込みました）")

        # 6. 除外されたIDを表示
        if excluded_ids:
            print("\n除外された patient_id:")
            print(", ".join(excluded_ids))
        else:
            print("\n除外された patient_id はありませんでした。")

    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")


if __name__ == "__main__":
    filter_summary_by_id_list()