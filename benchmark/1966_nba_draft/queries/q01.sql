WITH knicks_players AS (
    SELECT * FROM w WHERE "previous team" = 'new york knicks'
), not_used AS (
    SELECT * FROM w WHERE player = 'john barnhill'
) SELECT * FROM knicks_players WHERE "years of nba experience" > 1
AND {{
        LLMMap(
            'Did they play more than one position?',
            'knicks_players::pos'
        )
    }} = TRUE
