try:
    from .yolo_client import YoloClient
except Exception:  # pragma: no cover - optional runtime dependency (e.g., PIL)
    YoloClient = None

try:
    from .llm_controller import LLMController
except Exception:  # pragma: no cover - optional runtime dependency graph for tests
    LLMController = None

from .skillset import SkillSet, SkillItem, SkillArg
