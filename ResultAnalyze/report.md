# PhenoPacket difficult-case reanalysis

Fresh responses were obtained for the 15 requested cases using the original
HPO/image inputs. The target diagnoses are OMIM:618164 for the TRAF7 cases and
OMIM:618505 for the KDM6B cases.

`>100` means that the target was not found in that tool's 100-candidate probe.
`error` means that the tool did not return a response.

| Patient | GestaltMatcher | PubCaseFinder | Vector similarity | PhenoBrain | GPT top-1 |
|---|---:|---:|---:|---:|---|
| 11702 | 91 | error | 54 | 71 | KAT6B-related disorder |
| 11706 | 19 | error | >100 | >100 | Loeys-Dietz syndrome |
| 11710 | 20 | error | 32 | >100 | FLNA-related frontometaphyseal dysplasia |
| 11715 | 58 | error | 14 | 94 | Kabuki syndrome 1 |
| 11718 | 46 | error | >100 | >100 | Say-Barber-Biesecker-Young-Simpson syndrome |
| 11721 | 54 | error | 25 | >100 | Aarskog-Scott syndrome |
| 12293 | >100 | error | >100 | >100 | Simpson-Golabi-Behmel syndrome 1 |
| 12300 | 7 | error | 83 | >100 | Au-Kline syndrome |
| 12289 | >100 | error | >100 | >100 | SETD2-related Luscan-Lumish syndrome |
| 12292 | 12 | error | 19 | >100 | Kleefstra syndrome |
| 12299 | 16 | error | 84 | 93 | Legius syndrome |
| 12291 | 11 | error | >100 | >100 | Kleefstra syndrome |
| 11700 | 5 | error | >100 | >100 | Congenital myotonic dystrophy type 1 |
| 11708 | 14 | error | >100 | >100 | Opitz G/BBB syndrome type I |
| 11714 | 7 | error | >100 | >100 | Bohring-Opitz syndrome |

## Interpretation

- GestaltMatcher returned the correct OMIM in 13/15 cases; the two absent cases
  were 12293 and 12289.
- Vector similarity returned the correct OMIM in 7/15 cases.
- PhenoBrain returned the correct OMIM in 3/15 cases.
- GPT returned five candidates for every case, but the exact target OMIM was
  not among the five candidates in any of the 15 cases. The complete GPT
  candidate lists are in each `cases/<patient_id>.json` file.
- PubCaseFinder could not be ranked in this run: all 15 requests timed out
  after the initial retries, and a separate 120-second retry for 11702 also
  timed out. This is recorded as a service error, not as a clinical miss.

The machine-readable source of this table is `summary.csv`; request metadata,
all returned candidates, HPO inputs, and per-tool errors are in `cases/`.
