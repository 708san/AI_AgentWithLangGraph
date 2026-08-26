# HPO Analysis

## Method

Primary analysis uses Phen2Disease-style information content plus MONDO exact disease normalization.
Phen2Disease counts OMIM, ORPHA, MONDO, and other database IDs as separate diseases; this analysis maps exact OMIM/ORPHA disease matches to MONDO to avoid double-counting the same disease concept.

IC(t) = -log2(N(t) / N_all). HPO annotations are propagated to all ancestors in the HPO DAG, and each disease is counted at most once per HPO term.

Only positive phenotypic abnormality annotations are used: `aspect = P`, descendants of `HP:0000118`, and annotations with `qualifier = NOT` are excluded.

Disease normalization uses exact MONDO mappings only. Unmapped diseases are retained as source-specific fallback IDs, and conflicted mappings are not silently merged.

## Sources

Generated at: 2026-08-21T07:24:46.298539+00:00
HPO release/version: http://purl.obolibrary.org/obo/hp/releases/2026-06-23/hp.json
MONDO release/version: http://purl.obolibrary.org/obo/mondo/releases/2026-08-04/mondo.owl

### hp.json
- source URL: https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.json
- checked at: 2026-08-21T07:24:37.436641+00:00
- local file modified/downloaded at: 2026-08-21T07:11:11.924796+00:00
- downloaded this run: False
- bytes: 23019454
- SHA-256: `3b646565695329aa399e937883c68d5d424d0df5eaab2f22baa0e08d44fdbe87`

### phenotype.hpoa
- source URL: https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa
- checked at: 2026-08-21T07:24:37.436641+00:00
- local file modified/downloaded at: 2026-08-21T07:11:59.799965+00:00
- downloaded this run: False
- bytes: 35672303
- SHA-256: `89004f85b253f980ffe84218d2c080665cbf67a57bbb322111d6a2db5eb31dff`

### mondo.json
- source URL: http://purl.obolibrary.org/obo/mondo.json
- checked at: 2026-08-21T07:24:37.436641+00:00
- local file modified/downloaded at: 2026-08-21T07:13:36.187454+00:00
- downloaded this run: False
- bytes: 107497863
- SHA-256: `7cf8f1df31185555a21f5ffaf36663ca420671a9bc234fc737eb9bfa977ecd60`

### mondo_exactmatch_omim.sssom.tsv
- source URL: http://purl.obolibrary.org/obo/mondo/mappings/mondo_exactmatch_omim.sssom.tsv
- checked at: 2026-08-21T07:24:37.436641+00:00
- local file modified/downloaded at: 2026-08-21T07:24:39.458276+00:00
- downloaded this run: True
- bytes: 1527246
- SHA-256: `7dad9b7df2729c06f0bbfdc134ac4bf9982385a22a1cafe966e6942274c221d5`

### mondo_exactmatch_orphanet.sssom.tsv
- source URL: http://purl.obolibrary.org/obo/mondo/mappings/mondo_exactmatch_orphanet.sssom.tsv
- checked at: 2026-08-21T07:24:37.436641+00:00
- local file modified/downloaded at: 2026-08-21T07:24:40.221448+00:00
- downloaded this run: True
- bytes: 1510677
- SHA-256: `2f1dd7fc078d02852acd3bf8212437583eafe983477a07080bc68cb2f7b11836`

## Summary

- total raw disease IDs: 12935
- total canonical disease concepts: 10814
- disease count reduction after MONDO normalization: 2121
- already MONDO IDs: 0
- mapped to MONDO: 12718
- unmapped diseases: 217
- mapping conflicts: 0
- total HPO terms in output: 19120
- annotated HPO terms: 12725
- unannotated HPO terms: 6395
- positive phenotypic annotation rows: 267062
- skipped NOT annotations: 727
- raw data validation: all required files exist, have non-zero size, JSON files were parsed successfully, and phenotype.hpoa header was detected.

## Required Examples

### HP:0000707 Abnormality of the nervous system
- MONDO-normalized disease_count: 7036
- MONDO-normalized IC: 0.6200728736100384
- Phen2Disease raw disease_count: 8458
- Phen2Disease raw IC: 0.6128915885433016

### HP:0001250 Seizure
- MONDO-normalized disease_count: 2657
- MONDO-normalized IC: 2.0250301285914927
- Phen2Disease raw disease_count: 3122
- Phen2Disease raw IC: 2.050737611594194

## Sanity Checks

- monotonic violation count: 0
- duplicate propagation example: `{"raw_disease_id": "OMIM:614102", "direct_parent_hpo": "HP:0002719", "direct_child_hpo": "HP:0002205", "parent_count_contains_disease_once": true}`
- merged OMIM/ORPHA examples: `[{"canonical_id": "MONDO:0009288", "raw_ids": ["OMIM:232220", "OMIM:232240", "ORPHA:79259"]}, {"canonical_id": "MONDO:0010535", "raw_ids": ["OMIM:301845", "ORPHA:113", "ORPHA:166113"]}, {"canonical_id": "MONDO:0010574", "raw_ids": ["OMIM:304340", "ORPHA:1568", "ORPHA:85329"]}]`

### IC comparison

```json
{
  "HP:0000118": {
    "name": "Phenotypic abnormality",
    "mondo_normalized": {
      "ic": -0.0,
      "disease_count": 10814,
      "fraction": 1.0
    },
    "phen2disease_raw": {
      "information_content": -0.0,
      "disease_count": 12935,
      "total_disease_count": 12935,
      "disease_fraction": 1.0
    }
  },
  "HP:0000707": {
    "name": "Abnormality of the nervous system",
    "mondo_normalized": {
      "ic": 0.6200728736100384,
      "disease_count": 7036,
      "fraction": 0.6506380617717773
    },
    "phen2disease_raw": {
      "information_content": 0.6128915885433016,
      "disease_count": 8458,
      "total_disease_count": 12935,
      "disease_fraction": 0.653884808658678
    }
  },
  "HP:0001250": {
    "name": "Seizure",
    "mondo_normalized": {
      "ic": 2.0250301285914927,
      "disease_count": 2657,
      "fraction": 0.24570001849454412
    },
    "phen2disease_raw": {
      "information_content": 2.050737611594194,
      "disease_count": 3122,
      "total_disease_count": 12935,
      "disease_fraction": 0.24136064940085042
    }
  }
}
```

### Lowest IC terms

```json
[
  {
    "hpo_id": "HP:0000118",
    "name": "Phenotypic abnormality",
    "information_content": -0.0,
    "disease_count": 10814
  },
  {
    "hpo_id": "HP:0000707",
    "name": "Abnormality of the nervous system",
    "information_content": 0.6200728736100384,
    "disease_count": 7036
  },
  {
    "hpo_id": "HP:0033127",
    "name": "Abnormality of the musculoskeletal system",
    "information_content": 0.6357410503459088,
    "disease_count": 6960
  },
  {
    "hpo_id": "HP:0012638",
    "name": "Abnormal nervous system physiology",
    "information_content": 0.765347984542486,
    "disease_count": 6362
  },
  {
    "hpo_id": "HP:0000924",
    "name": "Abnormality of the skeletal system",
    "information_content": 0.9478566120495056,
    "disease_count": 5606
  },
  {
    "hpo_id": "HP:0000152",
    "name": "Abnormality of head or neck",
    "information_content": 0.948628863851809,
    "disease_count": 5603
  },
  {
    "hpo_id": "HP:0000234",
    "name": "Abnormality of the head",
    "information_content": 0.9730378935169086,
    "disease_count": 5509
  },
  {
    "hpo_id": "HP:0011842",
    "name": "Abnormal skeletal morphology",
    "information_content": 0.9869847120578269,
    "disease_count": 5456
  },
  {
    "hpo_id": "HP:0000478",
    "name": "Abnormality of the eye",
    "information_content": 1.1271084071198385,
    "disease_count": 4951
  },
  {
    "hpo_id": "HP:0000271",
    "name": "Abnormality of the face",
    "information_content": 1.1449938912424031,
    "disease_count": 4890
  }
]
```

### Highest IC terms

```json
[
  {
    "hpo_id": "HP:0000052",
    "name": "Urethral atresia, male",
    "information_content": 13.400612641081999,
    "disease_count": 1
  },
  {
    "hpo_id": "HP:0001039",
    "name": "Atheroeruptive xanthoma",
    "information_content": 13.400612641081999,
    "disease_count": 1
  },
  {
    "hpo_id": "HP:0001301",
    "name": "Chronic sensorineural polyneuropathy",
    "information_content": 13.400612641081999,
    "disease_count": 1
  },
  {
    "hpo_id": "HP:0001449",
    "name": "Duplication of metatarsal bones",
    "information_content": 13.400612641081999,
    "disease_count": 1
  },
  {
    "hpo_id": "HP:0001459",
    "name": "1-3 toe syndactyly",
    "information_content": 13.400612641081999,
    "disease_count": 1
  },
  {
    "hpo_id": "HP:0001468",
    "name": "Aplasia/Hypoplasia involving the musculature of the upper arm",
    "information_content": 13.400612641081999,
    "disease_count": 1
  },
  {
    "hpo_id": "HP:0001691",
    "name": "Muscular subvalvular aortic stenosis",
    "information_content": 13.400612641081999,
    "disease_count": 1
  },
  {
    "hpo_id": "HP:0001775",
    "name": "Tarsal osteovalgus",
    "information_content": 13.400612641081999,
    "disease_count": 1
  },
  {
    "hpo_id": "HP:0001983",
    "name": "Reduced lymphocyte surface expression of CD43",
    "information_content": 13.400612641081999,
    "disease_count": 1
  },
  {
    "hpo_id": "HP:0002048",
    "name": "Renal cortical atrophy",
    "information_content": 13.400612641081999,
    "disease_count": 1
  }
]
```

## Re-run

```bash
cd HPO_analysis
python scripts/build_hpo_information.py
```
