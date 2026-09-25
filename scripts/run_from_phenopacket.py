import sys
import os
import json
import argparse
import time

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.agent_pipeline import RareDiseaseDiagnosisPipeline
from agent.utils.response_serializer import omim_curie, omim_number

def parse_phenopacket(file_path: str) -> dict:
    """
    Phenopacket JSONファイルを解析し、パイプラインの入力に必要な情報を抽出する。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        phenopacket = json.load(f)

    subject = phenopacket.get('subject', {})
    patient_id = subject.get('id', 'unknown_patient')
    sex = subject.get('sex', 'Unknown')

    present_hpo_list = []
    absent_hpo_list = []
    
    for feature in phenopacket.get('phenotypicFeatures', []):
        hpo_id = feature.get('type', {}).get('id')
        if not hpo_id:
            continue
        
        if feature.get('excluded', False):
            absent_hpo_list.append(hpo_id)
        else:
            present_hpo_list.append(hpo_id)

    onset = "Unknown"

    return {
        "patient_id": patient_id,
        "sex": sex,
        "onset": onset,
        "present_hpo_list": present_hpo_list,
        "absent_hpo_list": absent_hpo_list
    }

def format_final_diagnosis(final_diagnosis_obj) -> dict:
    """
    パイプラインから返されたfinalDiagnosisオブジェクトを整形された辞書に変換する。
    """
    if not final_diagnosis_obj:
        return {"ans": [], "reference": ""}

    if isinstance(final_diagnosis_obj, dict):
        raw_ans = final_diagnosis_obj.get("ans", [])
        top_level_reference = final_diagnosis_obj.get("reference", "")
    elif hasattr(final_diagnosis_obj, "ans"):
        raw_ans = final_diagnosis_obj.ans
        top_level_reference = getattr(final_diagnosis_obj, "reference", "")
    else:
        return {"ans": [], "reference": ""}
    
    ans_list = []
    for diag in raw_ans:
        get_value = diag.get if isinstance(diag, dict) else getattr
        original_omim_id = get_value("OMIM_id", get_value("omim_id", "N/A"))
        ans_list.append({
            "rank": get_value("rank", "N/A"),
            "disease_name": get_value("disease_name", "Unknown"),
            "omim_id": omim_number(original_omim_id),
            "omim_id_curie": omim_curie(original_omim_id),
            "description": get_value("description", ""),
            "reference": get_value("reference", get_value("references", [])) or [],
        })
    
    return {"ans": ans_list, "reference": top_level_reference}

def run_pipeline_from_phenopacket(
    phenopacket_path: str,
    model_name: str,
    image_path_arg: str = None,
    output_mode: str = 'file',
    use_phenobrain: bool = False,
    output_dir: str = None,
):
    """
    指定されたPhenopacketファイルから情報を読み込み、診断パイプラインを実行する。
    output_modeに応じて、ファイル保存またはデータ返却を行う。
    """
    start_time = time.time()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    try:
        patient_data = parse_phenopacket(phenopacket_path)
        patient_id = patient_data["patient_id"]
        if output_mode in ['file', 'print']:
            print(f"Phenopacket '{os.path.basename(phenopacket_path)}' を読み込みました (患者ID: {patient_id})")
    except Exception as e:
        print(f"エラー: Phenopacketファイルの解析に失敗しました: {e}")
        return None if output_mode == 'return' else 1

    result_file_path = None
    run_dir = output_dir or os.path.join(project_root, "run_outputs")
    if output_mode == 'file':
        os.makedirs(run_dir, exist_ok=True)
        result_file_path = os.path.join(run_dir, f"{patient_id}.json")
        if os.path.exists(result_file_path):
            print(f"結果ファイルが既に存在するため、処理をスキップします。")
            return 0

    # 画像パスの処理: 引数で指定された場合のみ使用する
    image_path = None
    if image_path_arg:
        if os.path.exists(image_path_arg):
            image_path = image_path_arg
            if output_mode in ['file', 'print']:
                print(f"画像ファイルを使用します: {image_path}")
        else:
            if output_mode in ['file', 'print']:
                print(f"警告: 指定された画像ファイルが見つかりません: {image_path_arg}")
    
    if output_mode in ['file', 'print']:
        print(f"診断パイプラインを実行します... (モデル: {model_name})")
    
    log_dir = os.path.join(run_dir, "logs")
    node_result_dir = os.path.join(run_dir, "node_results")
    os.makedirs(node_result_dir, exist_ok=True)
    os.environ["AGENT_RESULT_DIR"] = node_result_dir
    log_filename = f"{patient_id}_{model_name.replace('gpt-', '')}.log"

    pipeline = RareDiseaseDiagnosisPipeline(
        model_name=model_name,
        enable_log=(output_mode == 'file'), # ログファイル生成はfileモードの時のみとする
        log_filename=log_filename,
        log_dir=log_dir,
    )
    
    # verbose=Falseでパイプライン側のpretty_printを抑制
    final_state = pipeline.run(
        hpo_list=patient_data["present_hpo_list"],
        absent_hpo_list=patient_data["absent_hpo_list"],
        image_path=image_path,
        onset=patient_data["onset"],
        sex=patient_data["sex"],
        patient_id=patient_id,
        use_phenobrain=use_phenobrain,
        verbose=False,
        public_response=True,
    )

    public_state = final_state
    
    # 最終診断結果を整形
    final_diagnosis_data = format_final_diagnosis(public_state.get("finalDiagnosis"))

    if output_mode == 'file' and result_file_path:
        # 結果をファイルに保存
        try:
            with open(result_file_path, 'w', encoding='utf-8') as f:
                json.dump(final_diagnosis_data, f, indent=4, ensure_ascii=False)
            print(f"診断結果を保存しました: {result_file_path}")
        except Exception as e:
            print(f"結果ファイルの保存に失敗しました: {e}")

    if output_mode == 'return':
        # モジュールとして呼び出された場合は、整形したデータを返す
        return final_diagnosis_data
    
    # --- 以下、コマンドライン実行時の処理 (file または print モード) ---
    print("\n--- 最終診断結果 ---")
    if not final_diagnosis_data or not final_diagnosis_data.get("ans"):
        print("最終診断結果は得られませんでした。")
    else:
        for diag in final_diagnosis_data["ans"]:
            print(f"Rank {diag['rank']}: {diag['disease_name']} ({diag['omim_id']})")
            print(f"  Description: {diag['description']}")
            print("-" * 20)
        
        reference_text = final_diagnosis_data.get("reference")
        if reference_text:
            print("\n--- References ---")
            print(reference_text)

    elapsed = time.time() - start_time
    print(f"\n処理が完了しました。経過時間: {elapsed:.2f}秒")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the rare disease diagnosis pipeline using a Phenopacket JSON file.")
    parser.add_argument("--phenopacket", type=str, required=True, help="Path to the input Phenopacket JSON file.")
    parser.add_argument("--model", type=str, default="gpt-4o", choices=["gpt-4o", "gpt-5-1", "gpt-5-2"], help="The name of the model to use.")
    parser.add_argument("--image", type=str, default=None, help="Optional path to the patient's image file.")
    parser.add_argument("--output_mode", type=str, default="file", choices=["file", "print", "return"], help="Output mode: 'file' to save JSON, 'print' to print to stdout.")
    parser.add_argument("--use_phenobrain", action="store_true", help="Enable PhenoBrain explicitly (default: disabled).")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for the final result, logs, and node responses.")
    
    args = parser.parse_args()
    
    run_pipeline_from_phenopacket(
        args.phenopacket,
        args.model,
        args.image,
        output_mode=args.output_mode,
        use_phenobrain=args.use_phenobrain,
        output_dir=args.output_dir,
    )
