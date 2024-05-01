from blendsql._program import Program
from guidance import gen

BASE_SYSTEM_PROMPT = """
This is a hybrid question answering task. The goal of this task is to answer the question given a table (`w`) and corresponding passages (`documents`).
Be as succinct as possible in answering the given question, do not include explanation.
"""


class EndtoEndProgram(Program):
    def __call__(self, serialized_db: str, question: str, **kwargs):
        with self.systemcontext:
            self.model += BASE_SYSTEM_PROMPT
        with self.usercontext:
            self.model += f"Context:\n{serialized_db}\n\n"
            self.model += f"Question: {question}\n"
            self.model += f"Answer:\n"
        with self.assistantcontext:
            self.model += gen(name="result", **self.gen_kwargs)
        return self.model
