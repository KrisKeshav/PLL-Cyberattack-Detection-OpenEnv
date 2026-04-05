"""
Pydantic models for the PLL Cyberattack Detection OpenEnv.
Defines Observation, Action, Reward, and State schemas.
"""
import numpy as np
from typing import Annotated, Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator
 
# Exactly 20 floats — enforced at validation time, not just documented.
WindowList = Annotated[List[float], Field(min_length=20, max_length=20)]
 
# Exactly 3 floats for [va, vb, vc].
VoltageList = Annotated[List[float], Field(min_length=3, max_length=3)]
 
class Observation(BaseModel):
    vq_window: WindowList
    vd_window: WindowList
    omega_window: WindowList
    omega_deviation_window: WindowList
    raw_voltages: VoltageList
    task_id: int = Field(ge=0, le=2)
    step: int = Field(ge=0)
 
class Action(BaseModel):
    attack_detected: bool
    attack_type: int = Field(ge=0, le=4)
    confidence: float = Field(ge=0.0, le=1.0)
    protective_action: int = Field(ge=0, le=3)
 
class Reward(BaseModel):
    total: float
    detection_reward: float
    classification_bonus: float
    early_detection_bonus: float
    false_alarm_penalty: float
    lock_loss_penalty: float
 
class State(BaseModel):
    theta_true: float
    theta_hat: float
    omega_hat: float
    vq_integral: float
    attack_active: bool
    attack_type: int         # Integer ID of the current attack: 0=none, 1=sinusoidal, 2=ramp, 3=pulse, 4=stealthy.
    attack_params: Dict[str, Any]
    attack_start_step: int
    lock_lost: bool     # Whether the PLL has lost lock (|theta_err| > 5°). Task 2 only.
    step: int = Field(ge=0)
    episode_id: str
    task_id: int = Field(ge=0, le=2)
 
    @model_validator(mode="before")
    @classmethod
    def coerce_attack_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coerce numpy scalar types inside attack_params to native Python types.
        sample_*_params() casts with float()/int() but a future contributor
        may forget. This validator ensures JSON serialization never fails due
        to np.float32 / np.int64 / np.bool_ leaking into the params dict.
        """
        params = values.get("attack_params", {})
        if isinstance(params, dict):
            coerced = {}
            for k, v in params.items():
                if isinstance(v, np.floating):
                    coerced[k] = float(v)
                elif isinstance(v, np.integer):
                    coerced[k] = int(v)
                elif isinstance(v, np.bool_):
                    coerced[k] = bool(v)
                else:
                    coerced[k] = v
            values["attack_params"] = coerced
        return values
