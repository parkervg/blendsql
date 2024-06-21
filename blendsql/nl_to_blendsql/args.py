from typing import Collection, List, Union
from dataclasses import dataclass, field


@dataclass
class NLtoBlendSQLArgs:
    max_grammar_corrections: int = field(
        default=0,
        metadata={
            "help": "Optional int defining maximum CFG-guided correction steps to be taken. This is based on the method in https://arxiv.org/pdf/2305.19234."
        },
    )

    include_db_content_tables: Union[List[str], str] = field(
        default="all",
        metadata={
            "help": "Which database tables to add `num_serialized_rows` worth of content for in serialization."
        },
    )

    num_serialized_rows: int = field(
        default=3,
        metadata={
            "help": "How many example rows to include in serialization of database"
        },
    )

    use_tables: Collection[str] = field(
        default=None,
        metadata={"help": "Collection of tables to use in serialization to string"},
    )

    use_bridge_encoder: bool = field(
        default=True,
        metadata={
            "help": "Whether to use Bridge Content Encoder during input serialization"
        },
    )
