# This tool is responsible for searching information using disease names.
def wiki_search(disease_name: str):
    pass

def pubmed_search(disease_name: str):
    pass

def disease_search(disease_name: str):
    pass

def diseaseSearchForDiagnosis(tentativeDiagnosis):
    for diag in tentativeDiagnosis.ans:
        disease_search(diag.disease_name)
    pass