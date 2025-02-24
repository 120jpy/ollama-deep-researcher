import os
from dataclasses import dataclass, fields
from typing import Any, Optional
import logging
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass
from enum import Enum

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    GOOGLE = "googlesearch"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the research assistant."""
    fetch_full_page: bool = True  # Default to False
    user_question: bool = True  # Default to True
    max_web_research_loops: int = 3
    local_llm: str = "phi4:14b-q4_K_M"
    search_api: SearchAPI = SearchAPI.GOOGLE  # Default to TAVILY
    ollama_base_url: str = "http://host.docker.internal:11434/"

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
