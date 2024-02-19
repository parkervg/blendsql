"""
Contains programs for guidance.
https://github.com/guidance-ai/guidance
"""
from textwrap import dedent
from guidance.models import Model, OpenAIChat
from guidance import gen, user, system, assistant, select
from typing import List
from contextlib import nullcontext


def _get_contexts(model: Model):
    usercontext = nullcontext()
    systemcontext = nullcontext()
    assistantcontext = nullcontext()
    if isinstance(model, OpenAIChat):
        usercontext = user()
        systemcontext = system()
        assistantcontext = assistant()
    return (usercontext, systemcontext, assistantcontext)


class GuidanceProgram:
    def __new__(
        self,
        model: Model,
        question: str,
        gen_kwargs: dict = None,
        few_shot: bool = False,
        **kwargs,
    ):
        self.model = model
        self.question = question
        self.gen_kwargs = gen_kwargs if gen_kwargs is not None else {}
        self.few_shot = few_shot
        assert isinstance(
            self.model, Model
        ), f"GuidanceProgram needs a guidance.models.Model object!\nGot {type(self.model)}"
        self.usercontext, self.systemcontext, self.assistantcontext = _get_contexts(
            self.model
        )
        return self.__call__(self, **kwargs)

    def __call__(self, *args, **kwargs):
        pass


class MapProgram(GuidanceProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        values: List[str],
        sep: str,
        include_tf_disclaimer: bool = False,
        output_type: str = None,
        example_outputs: str = None,
        table_title: str = None,
        colname: str = None,
        **kwargs,
    ):
        with self.systemcontext:
            self.model += dedent(
                """Given a set of values from a database, answer the question row-by-row, in order."""
            )
            if include_tf_disclaimer:
                self.model += dedent(
                    " If the question can be answered with 'true' or 'false', select `t` for 'true' or `f` for 'false'."
                )
            self.model += dedent(
                f"""
            The answer should be a list separated by '{sep}', and have {len(values)} items in total.
            When you have given all {len(values)} answers, stop responding.
            If a given value has no appropriate answer, give '-' as a response.
            """
            )
        with self.usercontext:
            if self.few_shot:
                self.model += dedent(
                    """
                --- 
    
                The following values come from the column 'Home Town', in a table titled '2010\u201311 North Carolina Tar Heels men's basketball team'.
                Q: What state is this?
                Values:
                `Ames, IA`
                `Carrboro, NC`
                `Kinston, NC`
                `Encino, CA`
    
                Output type: string
                Here are some example outputs: `MA;CA;-;`
    
                A: IA;NC;NC;CA
    
                --- 
    
                The following values come from the column 'Penalties (P+P+S+S)', in a table titled 'Biathlon World Championships 2013 \u2013 Men's pursuit'.
                Q: Total penalty count?
                Values:
                `1 (0+0+0+1)`
                `10 (5+3+2+0)`
                `6 (2+2+2+0)`
    
                Output type: numeric
                Here are some example outputs: `9;-`
    
                A: 1;10;6
    
                ---
    
                The following values come from the column 'term', in a table titled 'Electoral district of Lachlan'.
                Q: how long did it last?
                Values:
                `1859–1864`
                `1864–1869`
                `1869–1880`
    
                Output type: numeric
    
                A: 5;5;11
    
                ---
    
                The following values come from the column 'Length of use', in a table titled 'Crest Whitestrips'.
                Q: Is the time less than a week?
                Values:
                `14 days`
                `10 days`
                `daily`
                `2 hours`
    
                Output type: boolean
                A: f;f;t;t
    
                ---
                """
                )
            if table_title:
                self.model += dedent(
                    f"The following values come from the column '{colname}', in a table titled '{table_title}'."
                )
            self.model += dedent(f"""Q: {self.question}\nValues:\n""")
            for value in values:
                self.model += f"`{value}`\n"
            if output_type:
                self.model += f"\nOutput type: {output_type}"
            if example_outputs:
                self.model += f"\nHere are some example outputs: {example_outputs}\n"
            self.model += "\nA:"
        with self.assistantcontext:
            self.model += gen(name="result", **self.gen_kwargs)
        return self.model


class QAProgram(GuidanceProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        serialized_db: str,
        options: List[str] = None,
        long_answer: bool = False,
        table_title: str = None,
        **kwargs,
    ):
        with self.systemcontext:
            self.model += "Answer the question for the table. "
            if long_answer:
                self.model += "Make the answer as concrete as possible, providing more context and reasoning using the entire table."
            else:
                self.model += "Keep the answers as short as possible, without leading context. For example, do not say 'The answer is 2', simply say '2'."
        with self.usercontext:
            self.model += f"Question: {self.question}"
            if table_title is not None:
                model = f"Table Description: {table_title}"
            model += f"\n\n {serialized_db}"
        with self.assistantcontext:
            if options is not None:
                model += select(options=[str(i) for i in options], name="result")
            else:
                model += gen(name="result", **self.gen_kwargs)
        return model


class ValidateProgram(GuidanceProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, serialized_db: str, table_title: str = None):
        with self.systemcontext:
            self.model += "You are a database expert in charge of validating a claim given a context. Given a claim and associated database context, you will respond 'true' if the claim is factual given the context, and 'false' if not."
        with self.usercontext:
            self.model += f"Claim: {self.question}"
            if table_title:
                self.model += f"\nTable Description: {table_title}"
            self.model += f"\n{serialized_db}\n\nAnswer:"
        with self.assistantcontext:
            self.model += select(options=["true", "false"], name="result")
        return self.model


class JoinProgram(GuidanceProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        join_criteria: str,
        left_values: List[str],
        right_values: List[str],
        sep: str,
        **kwargs,
    ):
        left_values = "\n".join(left_values)
        right_values = "\n".join(right_values)
        with self.systemcontext:
            self.model += "You are a database expert in charge of performing a modified `LEFT JOIN` operation. This `LEFT JOIN` is based on a semantic criteria given by the user."
            self.model += f"\nThe left and right value alignment should be separated by '{sep}', with each new `JOIN` alignment goin on a newline. If a given left value has no corresponding right value, give '-' as a response."
        with self.usercontext:
            if self.few_shot:
                self.model += dedent(
                    """
                Criteria: Join to same topics.
    
                Left Values: 
                joshua fields
                bob brown
                ron ryan
    
                Right Values: 
                ron ryan
                colby mules
                bob brown (ice hockey)
                josh fields (pitcher)
    
                Output: 
                joshua fields;josh fields (pitcher)
                bob brown;bob brown (ice hockey)
                ron ryan;ron ryan
    
                --- 
    
                Criteria: Align the fruit to their corresponding colors.
    
                Left Values: 
                apple
                banana
                blueberry
                orange
    
                Right Values: 
                blue
                yellow
                red
    
                Output: 
                apple;red
                banana;yellow
                blueberry;blue
                orange;-
    
                ---
                """
                )
            self.model += dedent(
                f"""
                Criteria: {join_criteria}

                Left Values: {left_values}

                Right Values: {right_values}

                Output:
                """
            )
        with self.assistantcontext:
            self.model += gen(name="result", **self.gen_kwargs)
        return self.model
