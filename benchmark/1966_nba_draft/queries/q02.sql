SELECT title, player FROM w JOIN documents d ON{{
    LLMJoin(
        w.player,
        d.title
    )
}} WHERE {{
    LLMMap(
       'How many years with the franchise?',
       "career with the franchise"
    )
}} > 5