SELECT "Name", "Description" FROM parks
WHERE {{
    LLMMap(
        'Does this location have park facilities?',
        context='parks::Description'
    )
}} = FALSE