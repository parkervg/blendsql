SELECT title, player FROM w JOIN {{
    LLMJoin(
        left_on='documents::title',
        right_on='w::player'
    )
}} WHERE {{
    LLMMap(
       'How many years with the franchise?',
       'w::career with the franchise'
    )
}} > 5