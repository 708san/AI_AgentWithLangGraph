# Absent HPO付きGPT再解析

## 実験条件

- モデル：`gpt-5-2`
- 対象：前回と同じ15症例
- 変更点：`use_absentHPO=True`
- 入力：present HPOに加えて、TSVに明示された absent HPOを提示
- それ以外のプロンプト、モデル、Top 5制約は前回と同じ

## 結果

正解疾患がTop 5に含まれた症例は **0/15** で、Absent HPOを追加しても今回の
1回の試行では改善しなかった。

ただし、出力候補は変化した。Absent HPOなしの候補集合との一致は平均2.0/5で、
15症例中11症例では第1候補も変わった。したがって、Absent HPOはGPTの判断に影響
しているが、正解疾患を適切に上位へ引き上げる効果は確認できなかった。

| 症例 | Absent HPOなしの第1候補 | Absent HPOありの第1候補 |
|---|---|---|
| 11700 | Congenital myotonic dystrophy type 1 | VACTERL-H |
| 11702 | KAT6B-related disorder | KAT6B-related genitopatellar syndrome |
| 11706 | Loeys-Dietz syndrome | Shprintzen-Goldberg syndrome |
| 11708 | Opitz G/BBB syndrome | Coffin-Siris syndrome 6 |
| 11710 | FLNA-related frontometaphyseal dysplasia | Aarskog-Scott syndrome |
| 11714 | Bohring-Opitz syndrome | KBG syndrome |
| 11715 | Kabuki syndrome | Kabuki syndrome |
| 11718 | KAT6B-related disorder | KAT6B-related disorder |
| 11721 | Aarskog-Scott syndrome | Aarskog-Scott syndrome |
| 12289 | SETD2-related Luscan-Lumish syndrome | Tatton-Brown-Rahman syndrome |
| 12291 | Kleefstra syndrome | KAT6B-related disorder |
| 12292 | Kleefstra syndrome | Kleefstra syndrome 2 |
| 12293 | Simpson-Golabi-Behmel syndrome | Simpson-Golabi-Behmel syndrome |
| 12299 | Legius syndrome | Neurofibromatosis type 1 |
| 12300 | Au-Kline syndrome | Au-Kline syndrome |

## なぜ改善しなかったか

### 1. TRAF7群の陰性所見が少ない

TRAF7群の多くは absent HPO が「Seizure」1項目です。11708だけは「Seizure」と
「Feeding difficulties」が absent ですが、これだけではKAT6B、Kabuki、Aarskog、
FLNAなどの競合候補を十分に排除できない。

### 2. KDM6B群の陰性所見は候補の除外には使えるが、KDM6Bを直接示さない

KDM6B群には「けいれんなし」「口蓋裂なし」「聴覚障害なし」「合指なし」などが
多く含まれる。しかしこれらは、正解疾患を積極的に支持する特徴ではない。GPTは
候補疾患を消去する方向には使えても、正解疾患を新たに想起する手掛かりにはできず、
結果として別の既知症候群へ移っただけと考えられる。

例えば12293では、Absent HPOを追加してもLGA＋目立つ鼻＋発達遅滞から、
Simpson-Golabi-Behmel、Beckwith-Wiedemann、Sotosなどの過成長症候群が残った。
12299ではcafe-au-lait＋自閉＋発達退行が強く、NF1が第1候補になった。

### 3. 同じプロンプトでもGPT-5-2の出力は変動する

11702は元々 absent HPO が0個なので、Absent HPOなし／ありでプロンプトが完全に
同一だった。それでも第1候補は、通常実行のKAT6B-related disorderから、再実行では
KAT6B-related genitopatellar syndromeへ変わった。

これは、今回の比較にモデル出力の確率的変動も含まれることを示す。したがって、
Absent HPOの効果を厳密に評価するには、同一プロンプトを複数回実行するか、モデルの
seed・temperatureを固定した比較が必要である。

## 今回の結論

Absent HPOはGPTの候補順位を変える情報ではあるが、今回の症例では正解疾患を推定
するための十分な情報にならなかった。理由は、陰性所見が正解疾患に固有ではなく、
正解疾患名・遺伝子とHPOの対応をGPTが想起するための情報が不足しているためである。

今回の結果は「Absent HPOは無意味」という結論ではなく、少なくとも現行の
ゼロショットTop 5プロンプトだけでは、Absent HPO追加による改善は確認できない、
という結論である。

## 保存先

- 集計：[summary.csv](./summary.csv)
- 全結果：[summary.json](./summary.json)
- 症例別のプロンプトとTop 5：[cases/](./cases/)
