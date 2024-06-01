SELECT date, rival, score, documents.content AS "Team Description" FROM w
    JOIN {{
        LLMJoin(
            left_on='documents::title',
            right_on='w::rival'
        )
    }} WHERE rival = 'nsw waratahs'