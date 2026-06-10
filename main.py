# save as: main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
import pickle
from typing import Dict, Any, List
from pydantic import BaseModel

app = FastAPI(
    title="Medical Pipeline Sync API (Direct PKL Upload)",
    description="API designed to accept raw NLP and CV .pkl files, unpickle them, and find mismatches."
)

ORGAN_MAP = {
    "Liver": "liver", "liver": "liver",
    "Spleen": "spleen", "spleen": "spleen",
    "R_Kidney": "right_kidney", "right_kidney": "right_kidney",
    "L_Kidney": "left_kidney", "left_kidney": "left_kidney",
}

def normalize_status(status: Any) -> str:
    status_str = str(status).lower().strip()
    if status_str in ["present", "normal", "exists", "exist"]: 
        return "present"
    if status_str in ["missing", "removed", "removed/missing", "absent", "not present"]: 
        return "removed"
    return status_str

def extract_and_normalize(data: Any) -> Dict[str, str]:
    """
    Safely extracts the dictionary whether it's direct (NLP style)
    or nested under a Patient ID key (CV style).
    """
    if not isinstance(data, dict):
        raise ValueError("Unpickled data is not a valid dictionary.")

    # If it's nested like {"10": {"Liver": "present"}}, grab the inner dict
    if len(data) == 1 and not any(k.lower() in ORGAN_MAP for k in data.keys()):
        first_key = list(data.keys())[0]
        if isinstance(data[first_key], dict):
            data = data[first_key]

    normalized = {}
    for organ, value in data.items():
        clean_organ = ORGAN_MAP.get(organ, organ.lower().strip())
        normalized[clean_organ] = normalize_status(value)
    return normalized

# Pydantic schemas for the web response
class OrganComparisonResult(BaseModel):
    organ: str
    nlp_status: str
    cv_status: str
    match: bool

class CompareResponse(BaseModel):
    pid: str
    matches_all: bool
    summary: Dict[str, int]
    mismatches: List[OrganComparisonResult]
    details: List[OrganComparisonResult]


@app.post("/compare-pkl-files", response_model=CompareResponse)
async def compare_pkl_files(
    pid: str, 
    nlp_file: UploadFile = File(...), 
    cv_file: UploadFile = File(...)
):
    """
    Upload 11.pkl (as nlp_file) and 10.pkl (as cv_file) directly here.
    """
    try:
        # Read and unpickle the NLP file (e.g., 11.pkl)
        nlp_bytes = await nlp_file.read()
        raw_nlp_data = pickle.loads(nlp_bytes)
        
        # Read and unpickle the CV file (e.g., 10.pkl)
        cv_bytes = await cv_file.read()
        raw_cv_data = pickle.loads(cv_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Error reading or parsing the binary .pkl files: {str(e)}"
        )

    try:
        # Process and normalize both files
        nlp_out = extract_and_normalize(raw_nlp_data)
        cv_out = extract_and_normalize(raw_cv_data)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Data structure mismatch during normalization: {str(e)}"
        )
        
    all_organs = sorted(set(nlp_out.keys()) | set(cv_out.keys()))
    
    details, mismatches = [], []
    match_count, mismatch_count = 0, 0
    
    for organ in all_organs:
        nlp_status = nlp_out.get(organ, "N/A")
        cv_status = cv_out.get(organ, "N/A")
        is_match = (nlp_status == cv_status)
        
        item = OrganComparisonResult(organ=organ, nlp_status=nlp_status, cv_status=cv_status, match=is_match)
        details.append(item)
        if is_match: 
            match_count += 1
        else: 
            mismatch_count += 1
            mismatches.append(item)
            
    return CompareResponse(
        pid=pid,
        matches_all=(mismatch_count == 0),
        summary={"total_checked_organs": len(all_organs), "matches": match_count, "mismatches": mismatch_count},
        mismatches=mismatches,
        details=details
    )