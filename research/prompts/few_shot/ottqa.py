blendsql_examples = [
    {
        "serialized_db": 'CREATE TABLE "./List of cavalry recipients of the Victoria Cross (0)" (\n"name" TEXT,\n  "regiment" TEXT,\n  "date" TEXT,\n  "conflict" TEXT,\n  "location" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of cavalry recipients of the Victoria Cross (0)" LIMIT 3\n            name                                regiment      date        conflict      location\n herman albrecht                    imperial light horse  1900-1-6 second boer war     ladysmith\ncharles anderson    2nd dragoon guards ( queen \'s bays ) 1858-10-8   indian mutiny sundeela oudh\n  william bankes 7th ( the queen \'s own ) light dragoons 1858-3-19   indian mutiny       lucknow\n*/\n\nCREATE TABLE "./List of medical recipients of the Victoria Cross (0)" (\n"name" TEXT,\n  "regiment/corps" TEXT,\n  "date" TEXT,\n  "conflict" TEXT,\n  "location" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of medical recipients of the Victoria Cross (0)" LIMIT 3\n          name           regiment/corps                         date            conflict      location\nharold ackroyd royal berkshire regiment (xxxx-7-311917-8-1,p-38715d)     first world war passchendaele\n william allen          royal artillery                     1916-9-3     first world war   near mesnil\n henry andrews  indian medical services                   1919-10-22 waziristan campaign  khajuri post\n*/\n\nCREATE TABLE "./List of Royal Engineers recipients of the Victoria Cross (0)" (\n"name" TEXT,\n  "date" TEXT,\n  "conflict" TEXT,\n  "location" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of Royal Engineers recipients of the Victoria Cross (0)" LIMIT 3\n           name      date                conflict          location\n adam archibald 1918-11-4         first world war sambre-oise canal\n  fenton aylmer 1891-12-2    hunza-nagar campaign              nilt\nmark sever bell  1874-2-4 third anglo-ashanti war           ordashu\n*/\n\nCREATE TABLE "./List of Brigade of Guards recipients of the Victoria Cross (0)" (\n"name" TEXT,\n  "regiment" TEXT,\n  "date" TEXT,\n  "conflict" TEXT,\n  "location" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of Brigade of Guards recipients of the Victoria Cross (0)" LIMIT 3\n          name         regiment      date        conflict         location\n alfred ablett grenadier guards  1855-9-2     crimean war       sevastopol\njames ashworth grenadier guards 2012-6-13     afghanistan helmand province\n edward barber grenadier guards 1915-3-12 first world war   neuve chapelle\n*/\n\nCREATE TABLE "./List of Brigade of Gurkhas recipients of the Victoria Cross (0)" (\n"name" TEXT,\n  "unit" TEXT,\n  "date of action" TEXT,\n  "conflict" TEXT,\n  "place of action" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of Brigade of Gurkhas recipients of the Victoria Cross (0)" LIMIT 3\n            name                                                                       unit date of action                 conflict           place of action\n     john tytler 1 66th bengal native infantry later 1st king george v \'s own gurkha rifles           1858 indian rebellion of 1857 india choorpoorah , india\ndonald macintyre  2 bengal staff corps attached to 2nd king edward vii \'s own gurkha rifles           1872       looshai expedition   india lalgnoora , india\n  george channer    1 bengal staff corps attached to 1st king george v \'s own gurkha rifles           1875                perak war     malaya perak , malaya\n*/',
        "bridge_hints": "",
        "question": "What nationality was the winner of the 1945 award of the Victoria Cross ?",
        "blendsql": """
        {{
            LLMQA(
                "What was the recipient's nationality?",
                (
                    SELECT title AS 'Recipient', content FROM documents WHERE documents MATCH (
                        SELECT name || ' OR victoria cross'
                            FROM "./List of medical recipients of the Victoria Cross (0)"
                            WHERE SUBSTR(date, 0, 5) = '1945'
                    ) ORDER BY rank LIMIT 1
                )
            )
        }}
        """,
    },
    {
        "serialized_db": "",
        "bridge_hints": "",
        "question": "Which NHL team has the Player of the Year of Atlantic Hockey for the season ending in 2019 signed a agreement with ?",
        "blendsql": """
        {{
            LLMQA(
                "Which team has the player signed with?",
                (
                    SELECT * FROM documents WHERE documents MATCH (
                        SELECT name || ' OR hockey' FROM (
                            SELECT * FROM "./Atlantic Hockey Player of the Year (0)"
                            UNION ALL SELECT * FROM "./Atlantic Hockey Player of the Year (1)"
                        ) as w WHERE SUBSTR(w.year, -2) = '19' LIMIT 1
                    ) ORDER BY rank LIMIT 1
                )
            )
        }}
        """,
    },
    {
        "serialized_db": "",
        "bridge_hints": "",
        "question": "",
        "blendsql": """
        """,
    },
    # {
    #     "question": "How many points did Lebron James get in the NBA Season suspended by COVID-19?",
    #     "blendsql": """
    #     SELECT "Points Per Game" FROM "Lebron James Career Statistics"
    #     WHERE Year = {{
    #         LLMQA(
    #             'Which NBA season was suspended due to COVID-19?'
    #             (SELECT * FROM documents('nba OR covid') ORDER BY bm25(documents) LIMIT 1),
    #             options='w::Year'
    #         )
    #     }}
    #     """,
    # },
    # {
    #     "question": "When was the third highest paid Rangers F.C . player born ?",
    #     "blendsql": """
    #     {{
    #         LLMQA(
    #             'When was the player born?',
    #             (
    #                 SELECT title AS 'Player', content FROM documents
    #                     WHERE documents MATCH (
    #                         SELECT player FROM w
    #                             ORDER BY salary
    #                             LIMIT 1 OFFSET 3
    #                     )
    #                     ORDER BY bm25(documents) LIMIT 1
    #             )
    #         )
    #     }}
    #     """,
    # },
]
