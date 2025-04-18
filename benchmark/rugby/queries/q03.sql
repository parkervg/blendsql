SELECT date, rival, score, documents.content AS "Team Description" FROM w
    JOIN {{
        LLMJoin(
            'w::rival',
            'documents::title'
        )
    }} WHERE rival = 'nsw waratahs'