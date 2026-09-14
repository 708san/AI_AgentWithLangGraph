# CAFDADD / TRAF7 Phenopacket Tool-Group Analysis

Generated: 2026-09-08

## Question

同じ Phenopacket 正解疾患 `Cardiac, facial, and digital anomalies with developmental delay` (CAFDADD; TRAF7; OMIM:618164) の中で、ZebraSeek が答えられる症例、ZebraSeek は外すが PCF/GM が拾う症例、主要ツールがどれも拾えない症例の違いを調べる。

## Inputs

- Ranking overview: `local_artifacts/evaluation/validationCode/phenopacket_rank_overview_with_context/correct_disease_rank_overview_with_context.csv`
- Patient IC/category table: `local_artifacts/IC_eval/Patient_IC_eval/output/patient_ic_eval_table.csv`
- Raw phenopackets: `local_artifacts/evaluation/sampleData/0.1.25_filtered/TRAF7/*.json`
- ZebraSeek run JSON: `local_artifacts/run_outputs/res_5-2/*.json`
- HPO labels/categories: `HPO_analysis/output/hpo_information.json`

## Investigation Plan

1. OMIM:618164 の Phenopacket 症例だけを評価CSVから抽出する。
2. 評価CSVの `FinalDiagnosis`, `PubCaseFinder`, `GestaltMatcher` 判定を一次ソースとして A/B/C の3群へ分ける。
3. 患者IC表と raw phenopacket から、HPO数、IC、カテゴリ、cardiac/facial/digital/developmental anchor、急性新生児サインを症例別に付与する。
4. 実行結果JSONから tentative/final の候補名を取り、初期候補から最終候補への脱落を確認する。
5. 群別平均と HPO 出現率差をCSV化し、読み物用のMarkdownに要約する。

## Classification Rule

- A: ZebraSeek final diagnosis contains OMIM:618164 within the evaluated ranked answer.
- B: ZebraSeek final diagnosis misses OMIM:618164, but PubCaseFinder or GestaltMatcher contains OMIM:618164.
- C: ZebraSeek final diagnosis, PubCaseFinder, and GestaltMatcher all miss OMIM:618164.

## Summary

| Group | n | Patient IDs | Mean present HPO | Mean anchor count | Mean acute/neonatal count | PCF hit | GM hit |
|---|---:|---|---:|---:|---:|---:|---:|
| A_zebraseek_final_correct | 10 | 11701;11703;11704;11705;11707;11709;11711;11712;11713;11719 | 19.9 | 3.5 | 1.4 | 0.2 | 0.9 |
| B_initial_pcf_or_gm_correct_final_wrong | 3 | 11700;11708;11714 | 18 | 3.67 | 2.33 | 0.333 | 0.667 |
| C_no_major_tool_correct | 6 | 11702;11706;11710;11715;11718;11721 | 26.67 | 3.83 | 2 | 0 | 0 |

## Main Findings

1. ZebraSeek が最終的に正解した群は 10/19 例。GM が 9/10 例で先に拾っており、顔貌・眼周囲・発達/神経系の組み合わせが最終順位に残りやすい。
2. ZebraSeek は外したが PCF/GM が拾った群は 3/19 例。正解候補は初期候補または tentative に入っているが、最終 reranking で急性新生児像、骨格/筋緊張、広い多発奇形候補に押し出されている。
3. どれも拾えない群は 6/19 例。HPO数はむしろ多めで、単なる情報量不足ではない。症例ごとの表現が分散し、PCF/GM が使う典型的な CAFDADD アンカーへ接続できていない。
4. 疾患名の cardiac/facial/digital/developmental 4ドメインが揃うほど良い、という単純な構造ではない。失敗群にも cardiac/facial/digital/neuro の記載は存在するが、`feeding/hypotonia/respiratory/sepsis` などの非特異的な新生児重症サインが候補選択を強く引っ張る症例がある。

## Case Table

| Patient | Group | Eval final rank | JSON truth rank | PCF rank | GM rank | Anchors | Acute/neonatal | Top final candidates |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 11700 | B_initial_pcf_or_gm_correct_final_wrong | 999 | - | 999 | 2 | 3 | 5 | STUVE-WIEDEMANN SYNDROME 1; STWS1 \| GLYCINE ENCEPHALOPATHY WITH NORMAL SERUM GLYCINE \| NEURODEVELOPMENTAL DISORDER WITH NEONATAL RESPIRATORY INSUFFICIENCY, HYPOTONIA, AND FEEDING D |
| 11701 | A_zebraseek_final_correct | 5 | 2 | 999 | 999 | 3 | 0 | ReNU SYNDROME; RENU \| CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| KABUKI SYNDROME 2; KABUK2 \| PALLISTER W SYNDROME |
| 11702 | C_no_major_tool_correct | 999 | - | 999 | 999 | 3 | 2 | CHROMOSOME 14q11-q22 DELETION SYNDROME \| SNIJDERS BLOK-CAMPEAU SYNDROME; SNIBCPS \| PALLISTER W SYNDROME |
| 11703 | A_zebraseek_final_correct | 1 | 2 | 999 | 1 | 4 | 3 | WHITE-SUTTON SYNDROME; WHSUS \| CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| OHDO SYNDROME, SBBYS VARIANT; SBBYSS \| RITSCHER-SCHINZEL SYNDROME 3; RTSC3 |
| 11704 | A_zebraseek_final_correct | 5 | 5 | 999 | 5 | 3 | 1 | MULTIPLE CONGENITAL ANOMALIES-HYPOTONIA-SEIZURES SYNDROME 2; MCAHS2 \| NEURODEVELOPMENTAL DISORDER WITH CENTRAL HYPOTONIA AND DYSMORPHIC FACIES; NEDCHF \| CLEFT PALATE, PSYCHOMOTOR R |
| 11705 | A_zebraseek_final_correct | 1 | 1 | 1 | 1 | 3 | 1 | CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| OHDO SYNDROME, SBBYS VARIANT; SBBYSS \| WIEDEMANN-RAUTENSTRAUCH SYNDROME; WDRTS \| ReNU SYNDROME; RENU |
| 11706 | C_no_major_tool_correct | 999 | - | 999 | 999 | 4 | 0 | NOONAN SYNDROME 14; NS14 \| LOEYS-DIETZ SYNDROME 2; LDS2 \| KAUFMAN OCULOCEREBROFACIAL SYNDROME; KOS \| KBG SYNDROME; KBGS |
| 11707 | A_zebraseek_final_correct | 1 | 1 | 999 | 2 | 4 | 2 | CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| KABUKI SYNDROME 2; KABUK2 \| NOONAN SYNDROME 1; NS1 \| MULTIPLE CONGENITAL ANOMALIES-HYPOTONIA-SEIZURES SYN |
| 11708 | B_initial_pcf_or_gm_correct_final_wrong | 999 | 3 | 999 | 2 | 4 | 0 | OTOPALATODIGITAL SYNDROME, TYPE II; OPD2 \| CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| MUCOPOLYSACCHARIDOSIS, TYPE IVB; MPS4B \| ROBINOW SYNDROME, AUT |
| 11709 | A_zebraseek_final_correct | 2 | 2 | 999 | 3 | 4 | 2 | RITSCHER-SCHINZEL SYNDROME 3; RTSC3 \| CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| VERTEBRAL, CARDIAC, RENAL, AND LIMB DEFECTS SYNDROME 1; VCRL1 \| VAC |
| 11710 | C_no_major_tool_correct | 999 | - | 999 | 999 | 4 | 2 | CARDIOSPONDYLOCARPOFACIAL SYNDROME; CSCF \| RUBINSTEIN-TAYBI SYNDROME 1; RSTS1 \| KABUKI SYNDROME 2; KABUK2 \| OHDO SYNDROME, SBBYS VARIANT; SBBYSS |
| 11711 | A_zebraseek_final_correct | 1 | 1 | 2 | 3 | 4 | 2 | CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| CARDIOFACIONEURODEVELOPMENTAL SYNDROME; CFNDS \| RITSCHER-SCHINZEL SYNDROME 3; RTSC3 \| LOEYS-DIETZ SYNDROM |
| 11712 | A_zebraseek_final_correct | 4 | 3 | 999 | 4 | 3 | 1 | KAUFMAN OCULOCEREBROFACIAL SYNDROME; KOS \| CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| MENKE-HENNEKAM SYNDROME 1; MKHK1 |
| 11713 | A_zebraseek_final_correct | 1 | 1 | 999 | 1 | 3 | 2 | CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| ReNU SYNDROME; RENU \| GLYCINE ENCEPHALOPATHY WITH NORMAL SERUM GLYCINE \| NEURODEVELOPMENTAL DISORDER WITH |
| 11714 | B_initial_pcf_or_gm_correct_final_wrong | 999 | - | 5 | 999 | 4 | 2 | ReNU SYNDROME; RENU \| KLEEFSTRA SYNDROME 1; KLEFS1 \| ARBOLEDA-THAM SYNDROME; ARTHS \| YOU-HOOVER-FONG SYNDROME; YHFS \| MOWAT-WILSON SYNDROME; MOWS |
| 11715 | C_no_major_tool_correct | 999 | - | 999 | 999 | 4 | 2 | KABUKI SYNDROME 2; KABUK2 \| CARDIOSPONDYLOCARPOFACIAL SYNDROME; CSCF \| CHARGE SYNDROME \| CHROMOSOME 20q11-q12 DELETION SYNDROME |
| 11718 | C_no_major_tool_correct | 999 | - | 999 | 999 | 4 | 2 | OHDO SYNDROME, SBBYS VARIANT; SBBYSS \| ReNU SYNDROME; RENU \| COFFIN-SIRIS SYNDROME 1; CSS1 \| KABUKI SYNDROME 2; KABUK2 \| GLOBAL DEVELOPMENTAL DELAY, ABSENT OR HYPOPLASTIC CORPUS CA |
| 11719 | A_zebraseek_final_correct | 1 | 1 | 999 | 1 | 4 | 0 | CARDIAC, FACIAL, AND DIGITAL ANOMALIES WITH DEVELOPMENTAL DELAY; CAFDADD \| KABUKI SYNDROME 2; KABUK2 \| WARSAW BREAKAGE SYNDROME; WABS \| WITTEVEEN-KOLK SYNDROME; WITKOS |
| 11721 | C_no_major_tool_correct | 999 | - | 999 | 999 | 4 | 4 | AARSKOG SYNDROME, AUTOSOMAL DOMINANT \| LOEYS-DIETZ SYNDROME 2; LDS2 \| CONGENITAL HEART DEFECTS, DYSMORPHIC FACIAL FEATURES, AND INTELLECTUAL DEVELOPMENTAL DISORDER; CHDFIDD \| AU-KL |

## Data Quality Note

Primary grouping uses the evaluated ranking overview CSV. The current run JSON can differ from that snapshot:
- `11708`: evaluation CSV says final miss, but current run JSON contains OMIM:618164 at rank 3.

## Group Notes

### A_zebraseek_final_correct
- `11701`: FinalDiagnosis に正解が残った。 Present preview: Increased nuchal translucency | Global developmental delay | Delayed ability to walk | Periventricular leukomalacia | Ptosis | Epicanthus
- `11703`: FinalDiagnosis に正解が残った。 Present preview: Patent ductus arteriosus | Cerebellar vermis hypoplasia | Mask-like facies | Dolichocephaly | Narrow forehead | Protruding ear
- `11704`: FinalDiagnosis に正解が残った。 Present preview: Thickened nuchal skin fold | Stenosis of the external auditory canal | Patent ductus arteriosus | Gastroesophageal reflux | Delayed speech and language development | Widely spaced teeth
- `11705`: FinalDiagnosis に正解が残った。 Present preview: Hearing impairment | Short neck | Feeding difficulties | Hypertelorism | Low-set ears | Telecanthus
- `11707`: FinalDiagnosis に正解が残った。 Present preview: Hyperbilirubinemia | Short stature | Macrocephaly | Autism | Hypotonia | Ventriculomegaly
- `11709`: FinalDiagnosis に正解が残った。 Present preview: Patent ductus arteriosus | Feeding difficulties | Atrial septal defect | Hemivertebrae | Posteriorly rotated ears | Chiari malformation
- `11711`: FinalDiagnosis に正解が残った。 Present preview: Patent ductus arteriosus | Feeding difficulties | Microcephaly | Delayed ability to walk | Cerebral visual impairment | Optic disc pallor
- `11712`: FinalDiagnosis に正解が残った。 Present preview: Feeding difficulties | Delayed speech and language development | Delayed ability to walk | Blepharophimosis | Short palpebral fissure | Telecanthus
- `11713`: FinalDiagnosis に正解が残った。 Present preview: Cutis marmorata | Hamstring contractures | Epicanthus | Abnormality of visual evoked potentials | Seizure | Poor suck
- `11719`: FinalDiagnosis に正解が残った。 Present preview: Microcephaly | Delayed speech and language development | Delayed fine motor development | Simplified gyral pattern | Reduced cerebral white matter volume | Delayed CNS myelination

### B_initial_pcf_or_gm_correct_final_wrong
- `11700`: 初期候補では拾えた (GM rank 2 score 0.5007692307692307) が、final から落ちた。 Present preview: Short neck | Feeding difficulties | Lethargy | Torticollis | Short palpebral fissure | Hypotonia
- `11708`: 初期候補では拾えた (GM rank 2 score 0.4807692307692307) が、final から落ちた。 Present preview: Wide anterior fontanel | Atypical behavior | Delayed gross motor development | Dolichocephaly | Delayed closure of the anterior fontanelle | Depressed nasal bridge
- `11714`: 初期候補では拾えた (PCF rank 5 score 0.7969986242777789) が、final から落ちた。 Present preview: Long philtrum | Fetal ascites | Patent ductus arteriosus | Feeding difficulties | Delayed ability to walk | Neonatal respiratory distress

### C_no_major_tool_correct
- `11702`: FinalDiagnosis/PCF/GM のいずれにも正解が出ていない。 Present preview: Patent ductus arteriosus | Feeding difficulties | Inguinal hernia | Cerebral visual impairment | Axial hypotonia | Thin corpus callosum
- `11706`: FinalDiagnosis/PCF/GM のいずれにも正解が出ていない。 Present preview: Posteriorly rotated ears | Anteverted nares | Webbed neck | Pointed chin | Epicanthus | Blepharophimosis
- `11710`: FinalDiagnosis/PCF/GM のいずれにも正解が出ていない。 Present preview: Clinodactyly of the 2nd finger | Intestinal malrotation | Patent ductus arteriosus | Feeding difficulties | Arteria lusoria | Delayed ability to walk
- `11715`: FinalDiagnosis/PCF/GM のいずれにも正解が出ていない。 Present preview: Feeding difficulties | Posteriorly rotated ears | Epicanthus | Persistent left superior vena cava | Oral aversion | Renal malrotation
- `11718`: FinalDiagnosis/PCF/GM のいずれにも正解が出ていない。 Present preview: Tube feeding | Short stature | Delayed speech and language development | Dysarthria | Global developmental delay | Hypotonia
- `11721`: FinalDiagnosis/PCF/GM のいずれにも正解が出ていない。 Present preview: Dysphagia | Patent ductus arteriosus | Feeding difficulties | Atrial septal defect | Myopia | Syndactyly

## HPO Signals With Large Group Differences

| HPO | Name | Delta | A rate | B rate | C rate |
|---|---|---:|---:|---:|---:|
| HP:0000470 | Short neck | 0.833 | 0.5 | 1.0 | 0.167 |
| HP:0000316 | Hypertelorism | 0.833 | 0.5 | 0.0 | 0.833 |
| HP:0001263 | Global developmental delay | 0.833 | 0.4 | 0.0 | 0.833 |
| HP:0000581 | Blepharophimosis | 0.6 | 0.6 | 0.0 | 0.5 |
| HP:0000768 | Pectus carinatum | 0.567 | 0.1 | 0.333 | 0.667 |
| HP:0000028 | Cryptorchidism | 0.5 | 0.2 | 0.0 | 0.5 |
| HP:0000750 | Delayed speech and language development | 0.5 | 0.4 | 0.0 | 0.5 |
| HP:0000508 | Ptosis | 0.467 | 0.2 | 0.333 | 0.667 |
| HP:0001252 | Hypotonia | 0.433 | 0.6 | 0.333 | 0.167 |
| HP:0001159 | Syndactyly | 0.4 | 0.1 | 0.333 | 0.5 |
| HP:0001643 | Patent ductus arteriosus | 0.367 | 0.7 | 0.333 | 0.5 |
| HP:0012523 | Oral aversion | 0.333 | 0.0 | 0.0 | 0.333 |
| HP:0002680 | J-shaped sella turcica | 0.333 | 0.0 | 0.333 | 0.0 |
| HP:0001791 | Fetal ascites | 0.333 | 0.0 | 0.333 | 0.0 |
| HP:0005619 | Thoracolumbar kyphosis | 0.333 | 0.0 | 0.333 | 0.0 |
| HP:0000586 | Shallow orbits | 0.333 | 0.0 | 0.333 | 0.167 |
| HP:0000894 | Short clavicles | 0.333 | 0.0 | 0.333 | 0.0 |
| HP:0001476 | Delayed closure of the anterior fontanelle | 0.333 | 0.0 | 0.333 | 0.0 |
| HP:0000698 | Conical tooth | 0.333 | 0.1 | 0.333 | 0.0 |
| HP:0000034 | Hydrocele testis | 0.333 | 0.0 | 0.333 | 0.0 |
| HP:0002141 | Gait imbalance | 0.333 | 0.0 | 0.333 | 0.0 |
| HP:0009046 | Difficulty running | 0.333 | 0.0 | 0.333 | 0.0 |
| HP:0000003 | Multicystic kidney dysplasia | 0.333 | 0.0 | 0.333 | 0.0 |
| HP:0005487 | Prominent metopic ridge | 0.333 | 0.0 | 0.333 | 0.167 |
| HP:0002937 | Hemivertebrae | 0.333 | 0.1 | 0.333 | 0.0 |

## Next Checks

1. Final reranking に「PCF/GM top-5 に exact OMIM がある場合は少なくとも最終候補に保持する」ルールを入れ、11700/11708/11714 が救済されるか確認する。
2. CAFDADD の anchor HPO セットを disease-level recurrent HPO から作り、急性新生児サインだけで上位候補が置き換わるケースを監査する。
3. GM ヒットがあるのに final から落ちた症例は、Reflection/Final の negative evidence が「未記載」を「否定」に近く扱っていないか確認する。

## Output Files

- `case_group_table.csv`: 症例単位のツール順位、HPOドメイン、最終候補。
- `group_summary.csv`: 3群の平均値とヒット率。
- `hpo_group_signal.csv`: HPOごとの群別出現率。
