import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
JSON_FILE_PATH = ROOT_DIR / "historical_runs.json"



def log_run_info(run_id: str, run_name: str):
    if JSON_FILE_PATH.exists():
        with open(JSON_FILE_PATH, "r") as file:
            historical_runs = json.load(file)         
    else:
        historical_runs = []
        
    run_dict = {"run_id": run_id,
                "run_name": run_name}
    
    historical_runs.append(run_dict)
    
    with open(JSON_FILE_PATH, "w") as file:
        json.dump(historical_runs, file,indent=4)