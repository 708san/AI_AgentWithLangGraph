# 画像付き5症例ベンチマーク

既存の `ValidationDataWithoutDupli_newest.tsv` と `PhenoPacketStore_25072025` から、疾患の種類が異なる5症例を抽出した再実行用データです。

- 入力: present/absent HPO、性別、発症情報、顔画像へのパス
- 期待出力: 検証データに記録された正解疾患名と OMIM ID
- 実出力: エージェントが返した `finalDiagnosis`。期待出力と分けて保存します

## 収録症例

| 患者ID | 期待疾患 | OMIM | 画像 |
|---:|---|---|---|
| 272 | Cornelia de Lange syndrome 1 | OMIM:122470 | 355.jpg |
| 387 | White-Sutton syndrome | OMIM:616364 | 488.jpg |
| 11477 | Marfan lipodystrophy syndrome | OMIM:616914 | 19400.jpg |
| 11917 | Coffin-Siris syndrome 3 | OMIM:614608 | 20250.jpg |
| 13506 | Branchiooculofacial syndrome | OMIM:113620 | 24053.jpg |

詳細なHPOリストは [`data/five_cases.json`](../data/five_cases.json) にあります。

## 実行

まず入力と画像だけを検証する場合:

```bash
python scripts/run_five_cases.py --dry-run
```

Azure OpenAIを使って5件を実行する場合:

```bash
python scripts/run_five_cases.py --model gpt-5-2
```

出力は `run_outputs/five_cases/` に、症例ごとのJSONと全症例をまとめた `results.json` として保存されます。実行には既存の `.env` に指定モデル用のAzure OpenAI環境変数が必要です。
