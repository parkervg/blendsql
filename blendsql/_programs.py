"""
Contains programs for guidance.
https://github.com/guidance-ai/guidance/tree/e3c6fe93fa00cb86efc130bbce22aa29100936d4
"""

MAP_PROGRAM_COMPLETION = (
    lambda gen_clause: """
Given a set of values from a database, answer the question row-by-row, in order.
{{#if include_tf_disclaimer}}
If the question can be answered with 'true' or 'false', select `t` for 'true' or `f` for 'false'.
{{/if}}
The answer should be a list separated by '{{sep}}', and have {{answer_length}} items in total.
When you have given all {{answer_length}} answers, stop responding.
If a given value has no appropriate answer, give '-' as a response.

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

{{#if table_title}}
The following values come from the column '{{colname}}', in a table titled '{{table_title}}'.
{{/if}}
Q: {{question}}
Values:
{{~#each values_dict}}
`{{this.value}}`
{{/each}}

{{#if output_type}}
Output type: {{output_type}}
{{/if}}
{{#if example_outputs}}
Here are some example outputs: `{{example_outputs}}`
{{/if}}
A: """
    + gen_clause
)


MAP_PROGRAM_CHAT = (
    lambda gen_clause: """
{{#system~}}
Given a set of values from a database, answer the question row-by-row, in order.
{{#if include_tf_disclaimer}}
If the question can be answered with 'true' or 'false', select `t` for 'true' or `f` for 'false'.
{{/if}}
The answer should be a list separated by '{{sep}}', and have {{answer_length}} items in total.
When you have given all {{answer_length}} answers, stop responding.
If a given value has no appropriate answer, give '-' as a response.
{{~/system}}

{{#user~}}
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
`2 (0+1+1+0)`
`6 (2+2+2+0)`

Output type: numeric
Here are some example outputs: `9;-`

A: 1;2;6

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

{{#if table_title}}
The following values come from the column '{{colname}}', in a table titled '{{table_title}}'. 
Use this as context in responding.
{{/if}}
Q: {{question}}
Values:
{{~#each values_dict}}
`{{this.value}}`
{{/each}}
{{#if output_type}}
Output type: {{output_type}}
{{/if}}
{{#if example_outputs}}
Here are some example outputs: `{{example_outputs}}`
{{/if}}
A:
{{~/user}}

{{#assistant~}}
"""
    + gen_clause
    + "{{~/assistant}}"
)


# Need to make this a lambda since in guidance < 0.1,
#   `select` doesn't work with passing a `options` list directly
SELECT_PROGRAM_COMPLETION = (
    lambda options: """
Answer the given question by selecting exactly one of the possible values from a database.

Question: {{question}}
Values (separated by newline):
{{~#each values_dict}}
{{this.value}}
{{/each}}

{{select 'result' max_tokens=20 options="""
    + str(options)
    + "}}"
)


SELECT_PROGRAM_CHAT = (
    lambda options: """
{{#system~}}
Answer the given question by selecting exactly one of the possible values from a database
{{~/system}}

{{#user~}}
Question: {{question}}
Values (separated by newline):
{{~#each values_dict}}
{{this.value}}
{{/each}}
{{~/user}}

{{#assistant~}}
{{select 'result' max_tokens=20 options="""
    + str(options)
    + "}} {{~/assistant}}"
)


QA_PROGRAM_COMPLETION = (
    lambda gen_clause: """
Answer the question for the table.
{{#if long_answer}}
Make the answer as concrete as possible, providing more context and reasoning.
{{else}}
Keep the answers as short as possible, without leading context.
For example, do not say "The answer is 2", simply say "2".
{{/if}}

Question: {{question}}
{{#if table_title}}
Table Description: {{table_title}}
{{/if}}
{{serialized_db}}

{{#if options_dict}}
Select from one of the following options.
{{~#each options_dict}}
{{this.option}}
{{/each}}
{{/if}}

Answer: """
    + gen_clause
)


QA_PROGRAM_CHAT = (
    lambda gen_clause: """
{{#system~}}
Answer the question for the table. 
{{#if long_answer}}
Make the answer as concrete as possible, providing more context and reasoning using the entire table.
{{else}}
Keep the answers as short as possible, without leading context.
For example, do not say "The answer is 2", simply say "2".
{{/if}}
{{~/system}}

{{#user~}}
Question: {{question}}
{{#if table_title}}
Table Description: {{table_title}}
{{/if}}
{{serialized_db}}

{{#if options_dict}}
Select from one of the following options.
{{~#each options_dict}}
{{this.option}}
{{/each}}
{{/if}}

Answer:
{{~/user}}

{{#assistant~}}"""
    + gen_clause
    + "{{~/assistant}}"
)

JOIN_PROGRAM_CHAT = (
    lambda gen_clause: """
{{#system~}}
You are a database expert in charge of performing a modified `LEFT JOIN` operation.
This `LEFT JOIN` is based on a semantic criteria given by the user.
The left and right value alignment should be separated by '{{sep}}', with each new `JOIN` alignment goin on a newline.
If a given left value has no corresponding right value, give '-' as a response.
{{~/system}}

{{#user~}}
--- 

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
Criteria: {{join_criteria}}

Left Values: {{left_values}}

Right Values: {{right_values}}

Output:
{{~/user}}

{{#assistant~}}"""
    + gen_clause
    + "{{~/assistant}}"
)


JOIN_PROGRAM_COMPLETION = (
    lambda gen_clause: """
You are a database expert in charge of performing a modified `INNER JOIN` operation.
This `INNER JOIN` is based on a semantic criteria given by the user.
The left and right value alignment should be separated by '{{sep}}', with each new `JOIN` alignment goin on a newline.
If a given left value has no corresponding right value, give '-' as a response.

--- 

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
Criteria: {{join_criteria}}

Left Values: {{left_values}}

Right Values: {{right_values}}

Output:"""
    + gen_clause
)
