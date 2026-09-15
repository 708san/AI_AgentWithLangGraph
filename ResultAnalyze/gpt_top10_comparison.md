# GPT Top 10 比較

同じ15症例について、GPTに上位10疾患を出力させた。

- モデル：`gpt-5-2`
- 条件A：present HPOのみ
- 条件B：present HPO＋明示的なabsent HPO
- それ以外のプロンプトと症例入力は同じ

## 結果

両条件とも、正解疾患がTop 10に入った症例は **0/15** だった。

| 症例 | Absentなし Top 1 | Absentあり Top 1 |
|---|---|---|
| 11702 | KAT6B-related genitopatellar syndrome | Ohdo/SBBYS syndrome (KAT6B) |
| 11706 | Loeys-Dietz syndrome | Aarskog-Scott syndrome |
| 11710 | Aarskog-Scott syndrome | Loeys-Dietz syndrome type 3 |
| 11715 | Kabuki syndrome | Kabuki syndrome |
| 11718 | SBBYS syndrome (KAT6B) | SBBYS syndrome (KAT6B) |
| 11721 | Aarskog-Scott syndrome | Loeys-Dietz syndrome |
| 12293 | Malone-Benedict syndrome | Simpson-Golabi-Behmel syndrome |
| 12300 | KPTN-related disorder | Au-Kline syndrome |
| 12289 | Tatton-Brown-Rahman syndrome | Luscan-Lumish syndrome |
| 12292 | Kabuki syndrome | KBG syndrome |
| 12299 | NF1 | Legius syndrome |
| 12291 | Kleefstra syndrome | KAT6B-related disorder |
| 11700 | Cerebro-facio-thoracic dysplasia | Pallister-Killian syndrome |
| 11708 | Spondylocostal dysostosis | Kabuki syndrome |
| 11714 | Kleefstra syndrome | KBG syndrome |

## 解釈

Top 5からTop 10へ拡張しても、TRAF7（OMIM:618164）およびKDM6B
（OMIM:618505）の正解疾患は候補に入らなかった。追加された候補も、TRAF7群では
KAT6B、Kabuki、Aarskog、Loeys-Dietz、FLNAなど、KDM6B群ではSETD2、CHD8、
DNMT3A、PPP2R5D、KBGなど、表現型が似た既知疾患が中心だった。

これはTop 5制約だけが原因ではなく、正解疾患の想起自体が起きていないことを示す。
特にKDM6B群は発達遅滞・言語遅滞・低緊張・自閉傾向が中心で、Top 10にしても過成長・
自閉・クロマチン関連疾患の候補群に吸収された。

Absent HPOの追加によって順位は変化したが、除外情報が正解疾患を直接支持するわけでは
ないため、正解の順位を上げる効果は見られなかった。なお、`gpt-5-2`ではリポジトリの
ラッパーがtemperatureを固定していないため、Absent有無の比較は確率的な出力変動も
含む。厳密な差分評価には同じ条件を複数回実行する必要がある。

## 保存先

- present HPOのみ：[gpt_top10_present_only/summary.csv](./gpt_top10_present_only/summary.csv)
- Absent HPOあり：[gpt_top10_with_absent_hpo/summary.csv](./gpt_top10_with_absent_hpo/summary.csv)
- 各症例の全Top 10候補：それぞれの `cases/<patient_id>.json`
