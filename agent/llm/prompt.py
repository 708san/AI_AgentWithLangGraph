prompt_dict = {
    "diagnosis_prompt": (
        "you are a medical expert. Based on the following symptoms: {hpo_list}, "
        "and the top 5 disease candidates from PubCaseFinder: {top5}, "
         "and the top 5 disease candidates from Zero-Shot Diagnosis: {zeroShotResult}, "
        "please provide a final diagnosis for 5 top candidate disease."
    ),
    "zero-shot-diagnosis-prompt": (
         "You are a specialist in the field of rare diseases.\n"
        "You will be provided and asked about a complicated clinical case; read it carefully and then provide a "
        "diverse and comprehensive differential diagnosis.\n"
        "Patient’s {info_type}: {patient_info}\n"
        "Enumerate the top 5 most likely diagnoses. Be precise, and try to cover many unique possibilities.\n"
        "Each diagnosis should be a rare disease.\n"
        "Use ** to tag the disease name.\n"
        "Make sure to reorder the diagnoses from most likely to least likely.\n"
        "Now, List the top 5 diagnoses."
    )
}

def build_prompt(prompt_templete, inputs):
    return prompt_templete.format(**inputs)