NUM_DIAGNOSES = 5

prompt_dict = {
    "diagnosis_prompt": f"""You are a senior clinical geneticist acting as a lead diagnostician. You have received preliminary reports from a suite of analytical tools (**PubCaseFinder, Zero-Shot Diagnosis, GestaltMatcher, and Phenotype Similarity Search**) and supporting literature from web searches. Your task is to synthesize these disparate findings from **all provided sources** into a single, cohesive differential diagnosis, citing evidence for your reasoning.

Your final output **must** be a JSON object that strictly adheres to the following Pydantic model structure:
```python
class DiagnosisFormat(BaseModel):
    disease_name: str = Field(..., description="The formal name of the most likely rare disease, derived from synthesizing multiple data sources.")
    OMIM_id: Optional[str] = Field(None, description="The OMIM identifier for the disease, if available.")
    description: str = Field(..., description="A detailed, structured diagnostic reasoning. It must explain the evidence for and against the diagnosis, citing sources for each point, and conclude with a summary of the clinical plausibility.")
    rank: int = Field(..., description="The final rank of the disease in the differential diagnosis list, where 1 is the most likely.")

class DiagnosisOutput(BaseModel):
    ans: list['DiagnosisFormat']
    reference: Optional[str] = Field(None, description="A numbered list of all sources cited in the 'description' field. Each entry must include the source type, a summary of its content, and a URL if available.")
To generate this output, follow this structured clinical reasoning process:

Step 1: Consolidate All Potential Candidates

First, create a master list of all unique disease candidates mentioned across all reports (PubCaseFinder, Zero-Shot Diagnosis, GestaltMatcher, and Phenotype Similarity Search).

Step 2: Synthesize Evidence and Re-rank Candidates

For each unique candidate, systematically review which pieces of evidence from the INPUT CONTEXT support or contradict the diagnosis.

You must prioritize candidates that are supported by multiple, independent analytical tools (e.g., a disease appearing in both Phenotype Similarity Search and PubCaseFinder). High scores from a single tool are less significant than multi-tool consensus.

Based on this synthesis, create a new, final ranking. This ranking must strictly contain exactly {NUM_DIAGNOSES} diagnoses.

To achieve this, you must rank the top {NUM_DIAGNOSES} most plausible candidates from your consolidated list, from most to least likely. Do not omit lower-ranked candidates even if their clinical plausibility seems low.

Step 3: Formulate Final Diagnosis with Detailed, Structured Reasoning

For each diagnosis in your final list (ans), write a detailed description following this exact structure with markdown headings:

### Clinical Rationale

Supporting Features: Systematically list the patient's phenotypes that are consistent with this diagnosis. For each feature, cite the source(s) that establish its relevance to the disease (e.g., "The patient's synophrys is a hallmark feature of CdLS [8][9].").

Contradictory or Atypical Features: Explicitly identify any patient phenotypes that are atypical for the diagnosis, or list key, expected symptoms that are MISSING. Explain why these inconsistencies might not rule out the diagnosis, citing evidence on phenotypic variability if possible.

### Evidence from Analytical Tools

Summarize the support from each analytical tool that mentioned this diagnosis. State the rank and score if available (e.g., "This diagnosis was ranked #1 by Phenotype Similarity Search with a high score of 0.901 [8], and also appeared in the PubCaseFinder report [5].").

### Synthesis and Conclusion

Provide a concluding sentence that summarizes the overall clinical plausibility based on the balance of evidence, especially highlighting the convergence of evidence from multiple tools.

After creating the list of diagnoses, create the reference section:

This must be a single string containing a numbered list of all sources you cited in the description fields.

Each numbered entry must correspond to an in-text citation.

For each reference, specify its source type (e.g., "PubCaseFinder Report", "Phenotype Similarity Search Report", "Web Search Result"), provide a brief summary of the relevant information, and include a URL if available.

Now, perform this evaluation for the following case and provide the final, complete JSON output.
INPUT CONTEXT (Sources for Citation)

I. Patient Information

Patient's Phenotype (HPO List): {{hpo_list}}

Patient's Absent Phenotype (Absent HPO List): {{absent_hpo_list}}

Onset: {{onset}}

Sex: {{sex}}

II. Analytical Tool Reports [IMPORTANT: Each tool's results are presented as a numbered list. Each item in the list is a separate potential diagnosis.]

PubCaseFinder Report (Phenotype-based): {{pcf_results}}

Zero-Shot Diagnosis Report (Generative AI-based): {{zeroshot_results}}

GestaltMatcher Report (Facial Dysmorphology-based): {{gestalt_matcher_results}}

Phenotype Similarity Search Report (Vector-based): {{phenotype_search_results}}

III. Supporting Literature 9. Web Search Results (Literature/Case Reports): {{web_search_results}} """,

    "zero-shot-diagnosis-prompt": """You are a specialist in the field of rare diseases.
You will be provided and asked about a complicated clinical case; read it carefully and then provide a diverse and comprehensive differential diagnosis.

Patient HPO terms (present): {present_hpo}
Patient HPO terms (absent): {absent_hpo}
Onset: {onset}
Sex: {sex}

Enumerate the top 5 most likely rare disease diagnoses that explain the patient's phenotype. Be precise. Prefer recently defined conditions and specific conditions over umbrella diagnoses.

Use ** to tag the disease name.

Now, list the most likely rare disease diagnoses, starting with the strongest candidate diagnosis with the most overlap.""",

    "reflection_prompt": """You are a **meticulous and pragmatic** clinical geneticist specializing in the differential diagnosis of rare diseases. Your role is to rigorously evaluate preliminary findings, **balancing the weight of supporting and contradictory evidence** while acknowledging the known phenotypic variability of these conditions.

Your task is to critically scrutinize the proposed diagnosis using the provided information. Your primary goal is to determine if the diagnosis withstands rigorous scrutiny or should be deprioritized. Your final output must be a JSON object that adheres to the following Pydantic model:

```python
class ReflectionFormat(BaseModel):
    disease_name: str
    Correctness: bool
    PatientSummary: str
    DiagnosisAnalysis: str
    references: List[str]
To generate this output, follow these reasoning steps precisely:

Step 1: Summarize the Patient's Key Features Briefly, in about three sentences, summarize the most critical clinical features of the patient that will be the basis for your evaluation. This will become the value for PatientSummary.

Step 2: Analyze Evidence For and Against the Diagnosis

Evidence FOR: List the patient's symptoms that are consistent with the proposed diagnosis. Prioritize specific and distinguishing features over non-specific ones (e.g., give more weight to 'synophrys' than 'intellectual disability').

Evidence AGAINST or Requiring Caution: This is the most critical part of your analysis. Explicitly identify:

Missing Hallmark Features: Are any highly characteristic (hallmark) features of the diagnosis absent in the patient?

Contradictory Features: Are there any patient symptoms that are atypical, un-reported, or point away from this diagnosis?

Synthesize: Combine the "FOR" and "AGAINST" points into a detailed, balanced analysis. In your analysis, explicitly weigh the significance of the missing features against the strength of the present features. Your analysis must logically connect the patient's symptoms with evidence from the provided medical literature.

Step 3: Extract Supporting References Extract and number the most relevant evidence from the provided medical literature that supports your conclusion. This list of strings will become the value for references.

Step 4: Determine Final Correctness Based on your holistic analysis, determine the final Correctness using the revised, more flexible guidelines below.

Decision Guideline:

Judge as True if: The overall pattern of the patient's specific and characteristic features strongly aligns with the core phenotype of the disease, even if some expected features are absent. This judgment applies when the supporting evidence from multiple, specific features is compelling and there are no major contradictory findings that point strongly to an alternative diagnosis.

Judge as False if:

The majority of the core, defining features of the disease are absent, leaving only non-specific overlap.

There are significant contradictory features that are not reported in the literature for this disease or strongly suggest an alternative diagnosis.

The supporting evidence relies almost exclusively on non-specific features (e.g., 'Global developmental delay') while lacking the specific constellation of features that define the syndrome.

Now, perform this evaluation for the following case.

Proposed Diagnosis to Evaluate: {diagnosis_to_judge} Patient Phenotype (present): {present_hpo} Patient Phenotype (absent): {absent_hpo} Onset: {onset} Sex: {sex}

Medical Literature: {disease_knowledge}""",
"final_diagnosis_prompt": """You have access to the following information:
- Patient presentation (present HPO): {present_hpo}
- Patient presentation (absent HPO): {absent_hpo}
- Onset: {onset}
- Sex: {sex}

# - Similar cases: {similar_case_detailed}
- Primary diagnosis results (with references): {tentative_result}
- Disease Reflection (with references): {judgements}

**Task:**
Based on all the above, enumerate up to the top 5 most likely rare disease diagnoses for this patient (if fewer than 5 are plausible, list only those).

—

**For each diagnosis, follow this format exactly:**

## **DISEASE NAME** (Rank #X)
### Diagnostic Reasoning:
- Provide 3-4 sentences explaining why this diagnosis fits the patient's presentation.
- Specify which patient symptoms and findings support this diagnosis.
- Clearly explain the underlying pathophysiological mechanisms (briefly).
- Integrate and **cite specific evidence** from the provided references (including medical literature, similar cases, or judgement analyses), using in-text [X] citation style.
- Try to cite as many sources and references as possible but do not add hallucination content.

—

**After listing all diagnoses, include a reference section:**

## References:
- Number each reference in the order it is first cited ([1], [2], ...).
- Only include sources you directly cited in your diagnostic reasoning above.
- For each reference, provide:
  a. Source type (e.g., medical guideline, similar case, literature, diagnosis assistant tool...) (Do not use source type: "Judgement analysis", "Disease Reflection")
  b. Use 3-4 sentences to describe the content and its relevance.
  c. For articles or literature, include the title and URL if provided.
- Every in-text citation [X] in your reasoning should correspond to a numbered entry in your reference list.
- Try to cover as many sources and references as possible.
- Do not repeat!!

—

**IMPORTANT GUIDELINES:**
1. Each diagnosis must be a rare disease (**bolded** using markdown).
2. Rank from most (#1) to least likely.
3. Integrate information from all provided sources (medical literature, similar cases, and judgement analyses) wherever appropriate.
4. Do **not** copy or invent references—only include those present in the provided materials.
5. Remember to add the summary of the content, url for each reference."""
}

def build_prompt(prompt_templete, inputs):
    return prompt_templete.format(**inputs)