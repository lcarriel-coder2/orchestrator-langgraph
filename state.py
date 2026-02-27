from typing import TypedDict, List, Optional
from pydantic import BaseModel

class AgentState(TypedDict):
    # Identificadores básicos
    team_id: str
    input_prompt: str
    
    # IDs de seguimiento de BMAD
    current_epic_id: Optional[str]
    current_story_id: Optional[str]
    
    # Estado del código
    branch_name: Optional[str]
    commit_sha: Optional[str]
    pr_url: Optional[str]
    code_language: Optional[str]
    code_filepath: Optional[str]
    
    # Control de calidad y errores
    code_review_issues: List[dict]
    new_story_created: bool
    qa_results: dict
    failure_state: bool
    execution_log: List[str]


