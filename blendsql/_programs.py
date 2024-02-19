"""
Contains programs for guidance.
https://github.com/guidance-ai/guidance/tree/e3c6fe93fa00cb86efc130bbce22aa29100936d4
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


def map_program(
    model: Model,
    question: str,
    values: List[str],
    sep: str,
    include_tf_disclaimer: bool = False,
    output_type: str = None,
    example_outputs: str = None,
    table_title: str = None,
    colname: str = None,
    gen_kwargs: dict = None,
):
    usercontext, systemcontext, assistantcontext = _get_contexts(model)
    with systemcontext:
        model += dedent(
            """Given a set of values from a database, answer the question row-by-row, in order."""
        )
        if include_tf_disclaimer:
            model += dedent(
                "If the question can be answered with 'true' or 'false', select `t` for 'true' or `f` for 'false'."
            )
        model += dedent(
            f"""
        The answer should be a list separated by '{sep}', and have {len(values)} items in total.
        When you have given all {len(values)} answers, stop responding.
        If a given value has no appropriate answer, give '-' as a response.
        """
        )
    with usercontext:
        model += dedent(
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
            model += dedent(
                f"The following values come from the column '{colname}', in a table titled '{table_title}'."
            )
        model += dedent(f"""Q: {question}\nValues:\n""")
        for value in values:
            model += f"`{value}`\n"
        if output_type:
            model += f"Output type: {output_type}\n"
        if example_outputs:
            model += f"Here are some example outputs: {example_outputs}"
        model += "A:"
    with assistantcontext:
        model += gen(name="result", **gen_kwargs if gen_kwargs is not None else {})
    return model


def qa_program(
    model: Model,
    question: str,
    serialized_db: str,
    options: List[str] = None,
    long_answer: bool = False,
    table_title: str = None,
    gen_kwargs: dict = None,
):
    usercontext, systemcontext, assistantcontext = _get_contexts(model)
    with systemcontext:
        model += "Answer the question for the table. "
        if long_answer:
            model += "Make the answer as concrete as possible, providing more context and reasoning using the entire table."
        else:
            model += "Keep the answers as short as possible, without leading context. For example, do not say 'The answer is 2', simply say '2'."
    with usercontext:
        model += f"Question: {question}"
        if table_title is not None:
            model = f"Table Description: {table_title}"
        model += f"\n\n {serialized_db}"
    with assistantcontext:
        if options is not None:
            model += select(options=[str(i) for i in options], name="result")
        else:
            model += gen(name="result", **gen_kwargs if gen_kwargs is not None else {})
    return model


def validate_program(
    model: Model, claim: str, serialized_db: str, table_title: str = None
):
    usercontext, systemcontext, assistantcontext = _get_contexts(model)
    with systemcontext:
        model += "You are a database expert in charge of validating a claim given a context. Given a claim and associated database context, you will respond 'true' if the claim is factual given the context, and 'false' if not."
    with usercontext:
        model += f"Claim: {claim}"
        if table_title:
            model += f"\nTable Description: {table_title}"
        model += f"\n{serialized_db}\n\nAnswer:"
    with assistantcontext:
        model += select(options=["true", "false"], name="result")
    return model


def join_program(
    model: Model,
    join_criteria: str,
    left_values: List[str],
    right_values: List[str],
    sep: str,
    **gen_kwargs,
):
    usercontext, systemcontext, assistantcontext = _get_contexts(model)
    left_values = "\n".join(left_values)
    right_values = "\n".join(right_values)
    with systemcontext:
        model += "You are a database expert in charge of performing a modified `LEFT JOIN` operation. This `LEFT JOIN` is based on a semantic criteria given by the user."
        model += f"\nThe left and right value alignment should be separated by '{sep}', with each new `JOIN` alignment goin on a newline. If a given left value has no corresponding right value, give '-' as a response."
    with usercontext:
        model += dedent(
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
        model += dedent(
            f"""
        Criteria: {join_criteria}

        Left Values: {left_values}
        
        Right Values: {right_values}
        
        Output:
        """
        )
    with assistantcontext:
        model += gen(name="result", **gen_kwargs if gen_kwargs is not None else {})
    return model
