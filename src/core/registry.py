"""Live deployments: Deployment record + loaded predictors, shared by the agent and the API."""

from dataclasses import dataclass, field
from typing import Any

from core.contracts import Deployment, TaskSpec


@dataclass
class LiveDeployment:
    deployment: Deployment
    specs: dict[str, TaskSpec]
    predictors: dict[str, Any] = field(default_factory=dict)  # task_id -> Predictor


deployments: dict[str, LiveDeployment] = {}
