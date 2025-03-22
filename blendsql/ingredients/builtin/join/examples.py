from attr import attrs, attrib
from typing import List, Dict

from blendsql.utils import newline_dedent
from blendsql.ingredients.few_shot import Example


@attrs(kw_only=True)
class JoinExample(Example):
    join_criteria: str = attrib(default="Join to the same topics.")
    left_values: List[str] = attrib()
    right_values: List[str] = attrib()

    def to_string(self, *args, **kwargs) -> str:
        return newline_dedent(
            """Criteria: {}\n\nLeft Values:\n{}\n\nRight Values:\n{}\n\nOutput:""".format(
                self.join_criteria,
                "\n".join(self.left_values),
                "\n".join(self.right_values),
            )
        )


@attrs(kw_only=True)
class AnnotatedJoinExample(JoinExample):
    mapping: Dict[str, str] = attrib()
