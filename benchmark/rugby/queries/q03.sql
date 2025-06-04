SELECT date, rival, score, d.content AS "Team Description" FROM w
    JOIN documents d ON {{
        LLMJoin(
            w.rival,
            d.title
        )
    }} WHERE rival = 'nsw waratahs'