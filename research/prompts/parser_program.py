from blendsql._program import Program
from guidance import gen

BASE_SYSTEM_PROMPT = """
Generate BlendSQL given the question, table, and passages to answer the question correctly.
BlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.
These external functions should be wrapped in double curly brackets.

{ingredients_prompt}

Additionally, we have the table `documents` at our disposal, which contains Wikipedia articles providing more details about the values in our table.
ONLY use BlendSQL ingredients if necessary.
Answer parts of the question in vanilla SQL, if possible.
"""


class ParserProgram(Program):
    def __call__(
        self,
        ingredients_prompt: str,
        few_shot_prompt: str,
        serialized_db: str,
        question: str,
        bridge_hints: str = None,
        **kwargs,
    ):
        with self.systemcontext:
            self.model += BASE_SYSTEM_PROMPT.format(
                ingredients_prompt=ingredients_prompt
            )
        with self.usercontext:
            self.model += f"{few_shot_prompt}\n\n"
            self.model += f"{serialized_db}\n\n"
            if bridge_hints:
                self.model += (
                    f"Here are some values that may be useful: {bridge_hints}\n"
                )
            self.model += f"Q: {question}\n"
            self.model += f"BlendSQL:\n"
        print(self.model._current_prompt())
        with self.assistantcontext:
            self.model += gen(name="result", **self.gen_kwargs)
        return self.model
