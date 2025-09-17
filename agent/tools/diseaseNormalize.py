#this is a tool to mapping disease name to OMIM, using TogoSeek.
def disease_normalize(disease_name: str):
    pass

def diseaseNormalizeForDiagnosis(tentativeDiagnosis):
    for diag in tentativeDiagnosis.ans:
        disease_normalize(diag.disease_name)
    pass