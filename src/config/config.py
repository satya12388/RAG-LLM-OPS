from pathlib import Path
import yaml
from src.config.config_types import ParamsConfig

ROOT_PATH = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_PATH/"params.yaml"

def load_config_params(path:Path):

    with open(path,"r") as file:
        configs = yaml.safe_load(file)
        validated_config = ParamsConfig.model_validate(configs)

        return validated_config

params = load_config_params(path=CONFIG_PATH)
