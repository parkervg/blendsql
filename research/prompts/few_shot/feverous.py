# {
#     "serialized_db": "",
#     "bridge_hints": "",
#     "question": "",
#     "blendsql": """
#        """
# }
blendsql_examples = [
    {
        "serialized_db": 'CREATE TABLE "w0" (\n"index" INTEGER,\n  "attribute" TEXT,\n  "value" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index   attribute       value\n     0    kingdom:     plantae\n     1 (unranked): angiosperms\n     2 (unranked):    eudicots\n*/',
        "bridge_hints": "w0.attribute ( family: , order: )\nw0.value ( asteraceae , asterales )",
        "question": "Oyedaea is part of the family Asteraceae in the order Asterales.",
        "blendsql": """
        SELECT EXISTS (
            SELECT * FROM w0 WHERE attribute = 'family:' and value = 'asteraceae'
        ) AND EXISTS (
            SELECT * FROM w0 WHERE attribute = 'order:' and value = 'asterales'
        )
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w0" (\n"index" INTEGER,\n  "platform" INTEGER,\n  "line" TEXT,\n  "stopping pattern" TEXT,\n  "notes" REAL\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index  platform line                                    stopping pattern notes\n     0         1   t1   services to emu plains via central &amp; richmond  None\n     1         1   t9                 services to hornsby via strathfield  None\n     2         2   t1 terminating services to/from penrith &amp; richmond  None\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'unicode61 remove_diacritics 1\')',
        "bridge_hints": "documents.title ( lindfield railway station )",
        "question": "Lindfield railway station has 3 bus routes, in which the first platform services routes to Emu plains via Central and Richmond and Hornbys via Strathfield.",
        "blendsql": """
        SELECT EXISTS (
            SELECT * FROM w0 WHERE platform = 1 AND {{LLMMap('Does this service to Emu plains via Central and Richmond?', 'w0::stopping pattern')}} = TRUE
        ) AND EXISTS (
            SELECT * FROM w0 WHERE platform = 1 AND {{LLMMap('Does this service to Hornbys via Strathfield?', 'w0::stopping pattern')}} = TRUE
        ) AND EXISTS (
            SELECT * FROM docs WHERE {{LLMMap('How many bus routes operated by Transdev?', 'documents::content')}} = 3
        )
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w0" (\n"index" INTEGER,\n  "attribute" TEXT,\n  "value" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index            attribute                value\n     0 mukaradeeb \u0645\u0642\u0631 \u0627\u0644\u062f\u064a\u0628 mukaradeeb \u0645\u0642\u0631 \u0627\u0644\u062f\u064a\u0628\n     1              country                 iraq\n     2             province             al-anbar\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'unicode61 remove_diacritics 1\')',
        "bridge_hints": "w0.attribute ( district , province )\nw0.value ( al-anbar , al-qa'im )\ndocuments.title ( mukaradeeb )\ndocuments_content.c0 ( mukaradeeb )",
        "question": "Mukaradeeb('Wolf's Den') is a city in Iraq near the Syrian border, in the district of Al-Qa'im, province of Al-Anbar.",
        "blendsql": """
        SELECT EXISTS (
            SELECT * FROM docs WHERE {{LLMMap('Is it near the Syrian border?', 'documents::content')}} = TRUE
        ) AND EXISTS (
            SELECT * FROM w0 WHERE attribute = 'district' AND value = 'al-qa''im'
        ) AND EXISTS (
            SELECT * FROM w0 WHERE attribute = 'province' AND value = 'al-anbar'
        )
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w0" (\n"index" INTEGER,\n  "no." INTEGER,\n  "cr" INTEGER,\n  "filledcolumnname" TEXT,\n  "gp" INTEGER,\n  "w" INTEGER,\n  "l" INTEGER,\n  "otl" INTEGER,\n  "gf" INTEGER,\n  "ga" INTEGER,\n  "pts" INTEGER\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index  no.  cr filledcolumnname  gp  w  l  otl  gf  ga  pts\n     0    1   2    anaheim ducks  82 48 20   14 258 208  110\n     1    2   5  san jose sharks  82 51 26    5 258 199  107\n     2    3   6     dallas stars  82 50 25    7 226 197  107\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'unicode61 remove_diacritics 1\')',
        "bridge_hints": "w0.filledcolumnname ( san jose sharks )",
        "question": "The 2006-07 San Jose Sharks season, the 14th season of operation (13th season of play) for the National Hockey League (NHL) franchise, scored the most points in the Pacific Division.",
        "blendsql": """
        SELECT (
            {{
                LLMQA(
                    'Is the Sharks 2006-07 season the 14th season (13th season of play)?', 
                    'documents::content', 
                    options='t;f'
                )
            }} = 't'
        ) AND (
            SELECT (SELECT filledcolumnname FROM w0 ORDER BY pts DESC LIMIT 1) = 'san jose sharks'
        )
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w0" (\n"index" INTEGER,\n  "attribute" TEXT,\n  "value" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index   attribute                              value\n     0       motto business and technology - unlocked\n     1        type                            private\n     2 established                               1910\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'unicode61 remove_diacritics 1\')',
        "bridge_hints": "w0.attribute ( established , dean )\nw0.value ( rochester institute of technology , jacqueline r. mozrall )\ndocuments.title ( saunders college of business )\ndocuments_content.c0 ( saunders college of business )",
        "question": "Saunders College of Business, which is accredited by the Association to Advance Collegiate Schools of Business International, is one of the colleges of Rochester Institute of Technology established in 1910 and is currently under the supervision of Dean Jacqueline R. Mozrall.",
        "blendsql": """
        SELECT EXISTS(
            SELECT * FROM w0 WHERE attribute = 'parent institution' AND value = 'rochester institute of technology'
        ) AND EXISTS(
            SELECT * FROM w0 WHERE attribute = 'established' AND value = '1910'
        ) AND EXISTS(
            SELECT * FROM w0 WHERE attribute = 'dean' AND value = 'jacqueline r. mozrall'
        ) AND (
            {{
                LLMQA(
                    'Is Saunders College of Business (SCB) accredited by the Association to Advance Collegiate Schools of Business International (AACSB)?',
                    'documents::content',
                    options = 't;f'
                )
            }} = 't'
        )
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w0" (\n"index" INTEGER,\n  "0" TEXT,\n  "1" TEXT,\n  "2" TEXT,\n  "3" TEXT,\n  "4" TEXT,\n  "5" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index         0                      1                                 2                      3      4     5\n     0 candidate              candidate                             party               alliance  votes     %\n     1      none    mauricio vila dosal             national action party por yucata\u0301n al frente 447753  39.6\n     2      none mauricio sahui\u0301 rivero institutional revolutionary party     todos por yucata\u0301n 407802 36.09\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'unicode61 remove_diacritics 1\')',
        "bridge_hints": "w0.1 ( mauricio vila dosal )\nw0.2 ( national action party , party )\ndocuments.title ( 2018 mexican general election )\ndocuments_content.c0 ( 2018 mexican general election )",
        "question": "Mauricio Vila Dosal of the National Action Party overwhelmingly won the race for Governor of Yucatán during the 2018 Mexican general election.",
        "blendsql": """
        SELECT (
            {{
                LLMQA(
                    'Did Mauricio Vila Dosal of the National Action Party overwhelmingly win the race for Governor of Yucatán during the 2018 Mexican general election?',
                    (SELECT * FROM w0),
                    options='t;f'
                ) 
            }} = 't'
        )
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w0" (\n"index" INTEGER,\n  "attribute" TEXT,\n  "value" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index        attribute               value\n     0      preceded by freda meissner-blau\n     1     succeeded by          peter pilz\n     2 personal details    personal details\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'unicode61 remove_diacritics 1\')',
        "bridge_hints": "w0.attribute ( born )\ndocuments.title ( johannes voggenhuber )\ndocuments_content.c0 ( johannes voggenhuber )",
        "question": "Johannes Voggenhuber(born  5 June 1950) a politician who served as the former spokesman of the Green Party from Austria.",
        "blendsql": """
        SELECT (
            {{
                LLMQA(
                    'Was Johannes Voggenhuber born on 5 June 1950?',
                    (SELECT * FROM documents),
                    options='t;f'
                ) 
            }} = 't'
        ) AND (
            {{
                LLMQA(
                    'Was Johannes Voggenhuber former Member of the European Parliament (MEP) for the Austrian Green Party, which is part of the European Greens?',
                    (SELECT * FROM documents),
                    options='t;f'
                ) 
            }} = 't'
        )
        """,
    },
    {
        "serialized_db": "",
        "bridge_hints": "",
        "question": "Sixty two year old Welsh journalist Jan Moir worked for a couple other papers before working at Daily Mail as an opinion columnist and has won several awards for her writing.",
        "blendsql": """
        SELECT (
            SELECT {{LLMMap('What age?', 'w0::value')}} = 62 FROM w0 WHERE attribute = 'born'
        ) AND (
            {{
                LLMQA(
                    'Did Jan Moir work at a couple other papers before working at Daily Mail as an opinion columnist?',
                    (SELECT * FROM documents WHERE documents MATCH 'jan moir'),
                    options='t;f'
                ) 
            }} = 't'
        ) AND (
            {{
                LLMQA(
                    'Has Jan Moir won several awards for her writing?',
                    (SELECT * FROM documents WHERE documents MATCH 'jan moir'),
                    options='t;f'
                ) 
            }} = 't'
        )
        """,
    },
    {
        "serialized_db": "",
        "bridge_hints": "",
        "question": "Paspels use languages including German, and Romanish only and has recorded a total of 94.83% of German speakers in the 2000 census.",
        "blendsql": """
        SELECT NOT EXISTS (
            SELECT * FROM w0 WHERE languages NOT IN ('german', 'romanish')
        ) AND (
            SELECT percent = '94.38%' FROM w0 WHERE languages = 'german' AND census = 2000
        )
        """,
    },
    {
        "serialized_db": "",
        "bridge_hints": "",
        "question": "Retired Albanian football player Adrian Barbullushi never played for Egaleo or Ionikos.",
        "blendsql": """
        SELECT (
            SELECT {{
                LLMQA(
                    'Did Adrian Barbullushi play for Egaleo?',
                    (SELECT * FROM documents WHERE documents MATCH 'adrian barbullushi'),
                    options='t;f'
                )
            }} = 'f'
        ) AND (
            SELECT {{
                LLMQA(
                    'Did Adrian Barbullushi play for Ionikos?',
                    (SELECT * FROM documents WHERE documents MATCH 'adrian barbullushi'),
                    options='t;f'
                )
            }} = 'f'
        )
        """,
    },
]

sql_examples = []
