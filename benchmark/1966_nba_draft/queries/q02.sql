SELECT title, player FROM w JOIN {{
    LLMJoin(
        'w::player',
        'documents::title'
    )
}} WHERE {{
    LLMMap(
       'How many years with the franchise?',
       'w::career with the franchise'
    )
}} > 5