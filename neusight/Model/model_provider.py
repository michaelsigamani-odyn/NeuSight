import torch
from .model import ModelBase
import json

constructor = {}

# mlp wave models
from .mlp_wave_vec import MLPWaveVec
constructor["MLP_WAVE_VEC"] = MLPWaveVec
from .mlp_wave_mm import MLPWaveMM
constructor["MLP_WAVE_MM"] = MLPWaveMM

# habitat models
try:
    from .other.habitat_mm import HABITATMM
    constructor["HABITAT_MM"] = HABITATMM
    from .other.habitat_linear import HABITATLINEAR
    constructor["HABITAT_LINEAR"] = HABITATLINEAR
    from .other.habitat_vec import HABITATVEC
    constructor["HABITAT_VEC"] = HABITATVEC
    from .other.habitat_wave_mm import HabitatWaveMM
    constructor["HABITAT_WAVE_MM"] = HabitatWaveMM
    from .other.habitat_conv import HABITATConv
    constructor["HABITAT_Conv"] = HABITATConv
except:
    print("skipping importing habitat models")

# roofline models
try:
    from .other.roofline_mm import RooflineMM
    constructor["ROOFLINE_MM"] = RooflineMM
    from .other.roofline_vec import RooflineVEC
    constructor["ROOFLINE_VEC"] = RooflineVEC
except:
    print("skipping importing roofline models")

# micro models
try:
    from .other.micro_mm import MicroMM
    constructor["MICRO_MM"] = MicroMM
    from .other.micro_vec import MicroVEC
    constructor["MICRO_VEC"] = MicroVEC
except:
    print("skipping importing micro models")

# heuristics
try:
    from .other.heuristic_mm import HeuristicMM
    constructor["HEURISTIC_MM"] = HeuristicMM
    from .other.heuristic_vec import HeuristicVEC
    constructor["HEURISTIC_VEC"] = HeuristicVEC
except:
    print("skipping importing micro models")

def model_provider(config_path, tag=None, device=None) -> ModelBase:
    global constructor

    with open(str(config_path), "r") as f:
        config = json.load(f)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = constructor[config["architecture"]](config, tag, device)
    model.config["use_cache"] = False
    model = model.to(device)

    return model
