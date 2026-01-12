from dataclasses import dataclass, field

from blendsql.common.utils import newline_dedent
from blendsql.ingredients.few_shot import Example


@dataclass(kw_only=True)
class JoinExample(Example):
    join_criteria: str = field(default="Join to the same topics.")
    left_values: list[str] = field()
    right_values: list[str] = field()

    def to_string(self, *args, **kwargs) -> str:
        return newline_dedent(
            """Criteria: {}\n\nLeft Values:\n{}\n\nRight Values:\n{}\n\nOutput:""".format(
                self.join_criteria,
                "\n".join(self.left_values),
                "\n".join(self.right_values),
            )
        )


@dataclass(kw_only=True)
class AnnotatedJoinExample(JoinExample):
    mapping: dict[str, str] = field()
