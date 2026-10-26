"""The shapes (Pydantic models) every LLM reply must match. A reply that does not match
is sent back to the model once to be fixed.

Used by services/analyze.py and services/qa.py.
"""
from pydantic import BaseModel, Field, field_validator


class Summary(BaseModel):
    overview: str = Field(description="A short paragraph summarizing the meeting.")
    key_topics: list[str] = Field(description="The main topics discussed, a few words each.")


class Decision(BaseModel):
    topic: str = Field(description="2-4 word label of what was decided, e.g. 'launch date'. "
                                   "Used to merge pieces of long meetings.")
    text: str = Field(description="The final decision, in one sentence.")
    timestamp: float = Field(ge=0, description="Seconds: the t= value of the line where it was decided.")


class ActionItem(BaseModel):
    task: str = Field(description="What has to be done.")
    owner: str | None = Field(description="Person responsible, only if named in the transcript; else null.")
    due_date: str | None = Field(description="Deadline as stated (keep relative wording); else null.")
    timestamp: float = Field(ge=0, description="Seconds: the t= value of the line where it was assigned.")

    @field_validator("owner", "due_date")
    @classmethod
    def blank_is_null(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip()


class Extraction(BaseModel):
    decisions: list[Decision]
    action_items: list[ActionItem]
    open_topics: list[str] = Field(description="Topics discussed but not decided, as short labels.")


# ---------- merging the pieces of a long meeting (LLM proposes, code checks and applies) ----------

class Cluster(BaseModel):
    ids: list[str] = Field(description="Ids of items that are the same task / topic.")
    keep: str = Field(description="The id whose wording is clearest.")


class DecisionCluster(BaseModel):
    ids: list[str] = Field(description="Ids of decisions about the same subject.")


class DecidedTopic(BaseModel):
    open_id: str = Field(description="An open topic id.")
    decision_id: str = Field(description="The decision id that settles it.")


class MergePlan(BaseModel):
    action_clusters: list[Cluster]
    decision_clusters: list[DecisionCluster]
    open_topic_clusters: list[Cluster]
    decided_open_topics: list[DecidedTopic]
