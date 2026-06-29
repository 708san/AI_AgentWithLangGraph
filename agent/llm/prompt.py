NUM_DIAGNOSES = 10


def build_absent_hpo_section(
    absent_hpo,
    *,
    label: str,
    bullet: bool = False,
) -> str:
    """
    Build an absent-HPO section only when absent-HPO is actually used.

    Important:
    - If absent_hpo is None, empty, or not provided, return an empty string.
    - This prevents the prompt from showing an empty "Absent HPO" field.
    - Unreported findings should be treated as unknown, not absent.
    """
    if absent_hpo is None:
        return ""

    if isinstance(absent_hpo, str):
        absent_text = absent_hpo.strip()
    elif isinstance(absent_hpo, (list, tuple, set)):
        absent_text = ", ".join([str(x).strip() for x in absent_hpo if str(x).strip()])
    elif isinstance(absent_hpo, dict):
        absent_text = ", ".join([str(v).strip() for v in absent_hpo.values() if str(v).strip()])
    else:
        absent_text = str(absent_hpo).strip()

    if not absent_text:
        return ""

    prefix = "- " if bullet else ""
    return f"{prefix}{label}: {absent_text}"


prompt_dict = {
    "diagnosis_prompt_no_gestalt": """You are a senior clinical geneticist acting as a lead diagnostician.
**CRITICAL MISSION:** You must consolidate ALL potential diagnoses from the provided analytical tool reports into a single, comprehensive list.
**ABSOLUTE RULE:** DO NOT OMIT ANY CANDIDATE. Even if a disease appears only once with a low score, it MUST be included in the final output.

**Input Source:**
- Merged Candidate Table created from PubCaseFinder, Zero-Shot Diagnosis, and Phenotype Similarity Search.

**Note:** Facial image analysis (GestaltMatcher) is not available for this case.

**Task:**
1. Use the merged candidate table as the authoritative candidate list.
2. Do not split or duplicate candidates that have already been merged by OMIM ID or disease name.
3. Rank candidates based on multi-tool consensus, individual tool ranks, score strength, and clinical fit to the patient's HPO profile.
4. Output the result strictly following the format below.

**Strict Output Format Rules:**
- Do NOT use JSON, XML, or code blocks.
- Do NOT use markdown bolding (**), italics (*), or other styling.
- Each diagnosis candidate must be strictly enclosed within `===CASE_START===` and `===CASE_END===` lines.
- Each item must follow the `KEY::VALUE` format.
- The `DESCRIPTION` value must be a SINGLE LINE of text.
- After all diagnoses are listed, provide the references enclosed within `===REFERENCES_START===` and `===REFERENCES_END===`.

**Output Format Structure:**

===CASE_START===
RANK::[Integer]
DISEASE::[The formal name of the disease]
OMIM::[The OMIM identifier, or "N/A"]
DESCRIPTION::[A concise summary, max 2 sentences, stating WHY this disease is a candidate. Mention which tools supported it, such as "Supported by PCF score 0.9 and ZeroShot rank 1. Matches phenotype X, Y, Z."]
===CASE_END===

(Repeat for EVERY unique diagnosis found.)

===REFERENCES_START===
[A numbered list of all sources cited in the DESCRIPTION field.]
===REFERENCES_END===

---
INPUT CONTEXT

I. Patient Information
Phenotype: {hpo_list}
{absent_hpo_list_section}
Onset: {onset}
Sex: {sex}

II. Analytical Tool Reports
[Merged Candidate Table]:
{merged_candidate_results}

III. Web Search
{web_search_results}
""",

    "diagnosis_prompt": """You are a senior clinical geneticist acting as a lead diagnostician.
**CRITICAL MISSION:** You must consolidate ALL potential diagnoses from the provided analytical tool reports into a single, comprehensive list.
**ABSOLUTE RULE:** DO NOT OMIT ANY CANDIDATE. Even if a disease appears only once with a low score, it MUST be included in the final output.

**Input Source:**
- Merged Candidate Table created from PubCaseFinder, Zero-Shot Diagnosis, GestaltMatcher, and Phenotype Similarity Search.

**Task:**
1. Use the merged candidate table as the authoritative candidate list.
2. Do not split or duplicate candidates that have already been merged by OMIM ID or disease name.
3. Rank candidates based on multi-tool consensus, individual tool ranks, score strength, and clinical fit to the patient's HPO profile.
4. Output the result strictly following the format below.

**Strict Output Format Rules:**
- Do NOT use JSON, XML, or code blocks.
- Do NOT use markdown bolding (**), italics (*), or other styling.
- Each diagnosis candidate must be strictly enclosed within `===CASE_START===` and `===CASE_END===` lines.
- Each item must follow the `KEY::VALUE` format.
- The `DESCRIPTION` value must be a SINGLE LINE of text.
- After all diagnoses are listed, provide the references enclosed within `===REFERENCES_START===` and `===REFERENCES_END===`.

**Output Format Structure:**

===CASE_START===
RANK::[Integer]
DISEASE::[The formal name of the disease]
OMIM::[The OMIM identifier, or "N/A"]
DESCRIPTION::[A concise summary, max 2 sentences, stating WHY this disease is a candidate. Mention which tools supported it, such as "Supported by PCF score 0.9 and ZeroShot rank 1. Matches phenotype X, Y, Z."]
===CASE_END===

(Repeat for EVERY unique diagnosis found.)

===REFERENCES_START===
[A numbered list of all sources cited in the DESCRIPTION field.]
===REFERENCES_END===

---
INPUT CONTEXT

I. Patient Information
Phenotype: {hpo_list}
{absent_hpo_list_section}
Onset: {onset}
Sex: {sex}

II. Analytical Tool Reports
[Merged Candidate Table]:
{merged_candidate_results}

III. Web Search
{web_search_results}
""",

    "zero-shot-diagnosis-prompt": """You are a specialist in the field of rare diseases.
You will be provided and asked about a complicated clinical case. Read it carefully and provide a diverse and comprehensive differential diagnosis.

Patient HPO terms (present): {present_hpo}
{absent_hpo_section}
Onset: {onset}
Sex: {sex}

Important:
- If no absent HPO section is shown, treat unreported findings as unknown, not absent.
- Do not penalize a disease solely because a hallmark feature is not mentioned.
- Only use absent findings as negative evidence when they are explicitly provided above.

Enumerate the top 5 most likely rare disease diagnoses that explain the patient's phenotype.
Be precise. Prefer recently defined conditions and specific conditions over umbrella diagnoses.

Use ** to tag the disease name.

Now, list the most likely rare disease diagnoses, starting with the strongest candidate diagnosis with the most overlap.""",

    "reflection_prompt": """You are a meticulous and pragmatic clinical geneticist specializing in rare disease differential diagnosis.

Your task is to evaluate whether the proposed diagnosis remains sufficiently plausible to be retained as a final differential diagnosis candidate.

Important:
- You are NOT being asked whether the diagnosis is definitively confirmed.
- You are NOT being asked whether the diagnosis is molecularly proven.
- You are judging whether the diagnosis has enough clinical and phenotype-level support to remain in the final candidate list.
- Missing information should lower confidence only when appropriate. It should not automatically exclude a diagnosis.

Your final output must be a JSON object that adheres to the following Pydantic model:

```python
class ReflectionFormat(BaseModel):
    disease_name: str
    Correctness: bool
    PatientSummary: str
    DiagnosisAnalysis: str
    references: List[str]
````

Definition of Correctness:

* Correctness=True means the proposed diagnosis should be retained as a plausible final differential diagnosis candidate.
* Correctness=False means the proposed diagnosis should be excluded or strongly deprioritized because the available evidence is too weak, nonspecific, irrelevant, or contradictory.
* True does not mean the diagnosis is confirmed.
* False should not be assigned merely because molecular confirmation is unavailable.
* False should not be assigned merely because some hallmark features are unreported.

Treatment of missing and absent findings:

* Treat unreported findings as unknown, not absent.
* Only use absent findings as negative evidence when they are explicitly provided in the absent phenotype section.
* If no absent phenotype section is provided, do not infer absence of any clinical feature.
* A missing hallmark feature should reduce confidence, but it should not automatically make Correctness=False unless the remaining overlap is only nonspecific or the disease's core phenotype is clearly incompatible with the patient.

Treatment of disease identity and literature evidence:

* Use the proposed diagnosis name and its OMIM/MONDO identifier, when provided, as the disease identity anchor.
* Do not replace the proposed disease with a different gene, subtype, synonym, syndrome, or OMIM/MONDO entity unless the provided evidence clearly shows they are the same disease entity.
* If the retrieved medical literature appears to describe a different disease entity, gene, subtype, or only a loosely related syndrome, treat that as a retrieval or literature mismatch.
* Insufficient, missing, generic, or partially mismatched literature is not by itself evidence against the proposed diagnosis.
* If the provided literature does not cover the proposed OMIM/MONDO disease identity, decide retention primarily from the proposed diagnosis description, patient phenotype overlap, and diagnostic tool support.
* A high-ranked phenotype-based tool candidate should be retained unless there is a decisive clinical contradiction or the candidate's core phenotype is clearly incompatible with the patient.
* Apply a tool-supported candidate override: if PubCaseFinder, PhenotypeSearch, or GestaltMatcher ranks the proposed candidate highly and the patient has meaningful syndrome-level phenotype overlap, set Correctness=True unless there is a decisive contradiction.
* For newly described, sparsely documented, poorly retrieved, or otherwise literature-poor rare diseases, absence of detailed disease-specific literature should be treated as uncertainty rather than evidence against the disease.
* Assign Correctness=False because of literature only when the literature provides clear contradictory evidence, confirms a different disease identity, or shows that the candidate's defining phenotype is incompatible with the patient.

To generate this output, follow these steps.

Step 1: Summarize the patient's key features.
Briefly summarize the most important clinical features relevant to this proposed diagnosis. This will become the value for PatientSummary.

Step 2: Analyze evidence supporting retention of this candidate.
Explain which patient findings support the proposed diagnosis.
Prioritize specific and characteristic features over nonspecific ones.
First consider the Candidate Identity and Tool/Database Evidence section.
Then consider phenotype overlap.
Finally consider the Retrieved Medical Literature section.
Do not treat a weak or mismatched Retrieved Medical Literature section as negating the Candidate Identity and Tool/Database Evidence section.

Step 3: Analyze evidence that lowers confidence.
Identify missing, uncertain, atypical, or contradictory points.
Distinguish clearly between:

* features that are explicitly absent,
* features that are simply unreported,
* features that are truly contradictory,
* limitations caused by insufficient or mismatched literature.
  Do not over-penalize unreported hallmark features unless they are essential to the disease definition and the remaining overlap is weak.

Step 4: Synthesize the evidence.
Provide a balanced analysis explaining whether the candidate should remain in the final differential diagnosis list.
Explicitly state whether the limitations merely reduce confidence or are strong enough to exclude the candidate.
Your analysis must logically connect the patient's symptoms with evidence from the provided medical literature when the literature is relevant.
When the provided literature is insufficient or appears mismatched to the proposed OMIM/MONDO disease identity, explicitly state that limitation and do not treat it alone as a reason to exclude the candidate.

Step 5: Extract supporting references.
Extract the most relevant evidence from the provided medical literature.
Each reference string must explicitly include the source name and URL if available.
Format each reference strictly as:
"[Source Name] (URL): Extracted evidence text..."
Do not invent references.

Step 6: Determine final Correctness.

Judge as True if:

* The diagnosis explains multiple important findings in the patient.
* The overall phenotype is reasonably compatible with the disease.
* The candidate is supported by tool rankings, phenotype overlap, literature, or disease-specific knowledge.
* The provided literature is limited or partially mismatched, but the proposed disease identity remains plausible based on phenotype-level evidence and there is no decisive contradiction.
* The candidate is highly ranked by a phenotype-based diagnostic tool and the patient has meaningful phenotype-level overlap, even if disease-specific literature is unavailable or mismatched.
* The candidate is a specific, literature-poor rare disease with strong phenotype-based tool support and no decisive contradiction.
* There is no decisive contradiction.
* The diagnosis is reasonable to retain in the final differential diagnosis list, even if confirmation is still needed.

Judge as False if:

* The support relies almost entirely on nonspecific overlap.
* Relevant disease-specific evidence is unavailable and there is no meaningful phenotype-level or tool-ranking support.
* The disease's defining phenotype is largely incompatible with the patient.
* Explicitly absent findings directly contradict the disease's core phenotype.
* The proposed diagnosis is confirmed to be a different OMIM/MONDO disease entity from the evidence being evaluated.
* The candidate should not be carried into the final top diagnosis list.

Now evaluate the following case.

Candidate Identity and Tool/Database Evidence:
{diagnosis_to_judge}

{tool_support}

Patient Phenotype (present):
{present_hpo}
{absent_hpo_section}

Onset:
{onset}

Sex:
{sex}

Retrieved Medical Literature:
{disease_knowledge}
""",

    "final_diagnosis_prompt": """You have access to the following information:

* Patient presentation (present HPO): {present_hpo}
  {absent_hpo_bullet_section}
* Onset: {onset}
* Sex: {sex}

# - Similar cases: {similar_case_detailed}

* Primary diagnosis results (with references): {tentative_result}
* Disease Reflection (with references): {judgements}

Important interpretation of Disease Reflection:

* In Disease Reflection, Correctness means whether the disease remains plausible enough to be retained as a final differential diagnosis candidate.
* Correctness=True does not mean the disease is confirmed.
* Correctness=False usually means the disease should be deprioritized or excluded, but first inspect the stated reason.
* If Correctness=False is mainly due to insufficient literature, mismatched literature, or lack of disease-specific references, re-evaluate the candidate using the primary diagnosis result, OMIM/MONDO identity, phenotype overlap, and diagnostic tool support before excluding it.
* A highly ranked phenotype-based tool candidate should remain eligible for the final top 5 unless there is a decisive clinical contradiction or clear phenotype incompatibility.
* Do not prioritize a candidate merely because more literature was retrieved for it. Literature-rich alternatives should not displace a sparse but specific OMIM/MONDO candidate that has strong diagnostic-tool support and meaningful phenotype overlap.
* For newly described, sparsely documented, or poorly retrieved rare diseases, sparse literature should lower certainty but should not by itself lower rank below less tool-supported alternatives.
* Do not treat unreported findings as absent.
* Only use absent HPO findings as negative evidence when an absent HPO section is explicitly provided above.

**Task:**
Based on all the above, enumerate up to the top 5 most likely rare disease diagnoses for this patient.
If fewer than 5 are plausible, list only those.
Balance evidence in this order: phenotype specificity, diagnostic tool support, reflection reasoning, and relevant literature. Do not let literature volume alone dominate the ranking.

—

**For each diagnosis, follow this format exactly:**

## **DISEASE NAME** (Rank #X)

### Diagnostic Reasoning:

* Provide 3-4 sentences explaining why this diagnosis fits the patient's presentation.
* Specify which patient symptoms and findings support this diagnosis.
* Clearly explain the underlying pathophysiological mechanisms briefly.
* Integrate and cite specific evidence from the provided references, including medical literature, similar cases, or diagnostic tool outputs, using in-text [X] citation style.
* Try to cite as many relevant sources and references as possible, but do not add hallucinated content.

—

**After listing all diagnoses, include a reference section:**

## References:

* Number each reference in the order it is first cited: [1], [2], ...
* Only include sources you directly cited in your diagnostic reasoning above.
* For each reference, provide:
  a. Source type, such as medical guideline, similar case, literature, or diagnosis assistant tool.
  b. Use 3-4 sentences to describe the content and its relevance.
  c. For articles or literature, include the title and URL if provided.
* Do not use source type: "Judgement analysis" or "Disease Reflection".
* Every in-text citation [X] in your reasoning must correspond to a numbered entry in your reference list.
* Do not repeat references.

—

**IMPORTANT GUIDELINES:**

1. Each diagnosis must be a rare disease, bolded using markdown.
2. Rank from most likely (#1) to least likely.
3. Integrate information from all provided sources wherever appropriate.
4. Do not copy or invent references. Only include sources present in the provided materials.
5. Remember to add a summary of the content and URL for each reference when available.
""",
}

def build_prompt(prompt_templete, inputs):
    """
    Build prompts while removing absent-HPO sections when absent HPO is not used.

    Expected behavior:
    - If use_absentHPO is False, absent-HPO sections are removed from the prompt.
    - If use_absentHPO is True but no absent HPO terms are available, absent-HPO sections are also removed.
    - This prevents the model from interpreting an empty absent-HPO field as meaningful negative evidence.
    """
    use_absent_hpo = inputs.get("use_absentHPO", False)

    absent_hpo = (
        inputs.get("absent_hpo")
        or inputs.get("absent_hpo_list")
        or inputs.get("absentHpoDict")
        or ""
    )

    if use_absent_hpo:
        inputs = {
            **inputs,
            "absent_hpo_section": build_absent_hpo_section(
                absent_hpo,
                label="Patient HPO terms (absent)",
                bullet=False,
            ),
            "absent_hpo_list_section": build_absent_hpo_section(
                absent_hpo,
                label="Absent",
                bullet=False,
            ),
            "absent_hpo_bullet_section": build_absent_hpo_section(
                absent_hpo,
                label="Patient presentation (absent HPO)",
                bullet=True,
            ),
        }
    else:
        inputs = {
            **inputs,
            "absent_hpo_section": "",
            "absent_hpo_list_section": "",
            "absent_hpo_bullet_section": "",
        }

    return prompt_templete.format(**inputs)
