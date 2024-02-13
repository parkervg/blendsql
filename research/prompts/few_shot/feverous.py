# {
#     "serialized_db": "",
#     "bridge_hints": "",
#     "question": "",
#     "blendsql": """
#        """
# }
blendsql_examples = [
    {
        "serialized_db": 'Table Description: Oyedaea\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "oyedaea" TEXT,\n  "scientific classification" TEXT,\n  "kingdom:" TEXT,\n  "(unranked):" TEXT,\n  "(unranked):_2" TEXT,\n  "(unranked):_3" TEXT,\n  "order:" TEXT,\n  "family:" TEXT,\n  "tribe:" TEXT,\n  "genus:" TEXT,\n  "type species" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index oyedaea scientific classification kingdom: (unranked): (unranked):_2 (unranked):_3    order:    family:      tribe:      genus: type species\n     0 oyedaea scientific classification  plantae angiosperms      eudicots      asterids asterales asteraceae heliantheae oyedaea dc. type species\n*/',
        "bridge_hints": "w0.oyedaea ( oyedaea )\nw0.order: ( asterales )\nw0.family: ( asteraceae )",
        "question": "Oyedaea is part of the family Asteraceae in the order Asterales.",
        "blendsql": """
        SELECT EXISTS (
            SELECT * FROM w0 WHERE "family:" = 'asteraceae' AND "order:" = 'asterales'
        ) 
        """,
    },
    {
        "serialized_db": 'Table Description: Lindfield railway station\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "platform" INTEGER,\n  "line" TEXT,\n  "stopping pattern" TEXT,\n  "notes" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index  platform line                                    stopping pattern notes\n     0         1   t1   services to emu plains via central &amp; richmond notes\n     1         1   t9                 services to hornsby via strathfield notes\n     2         2   t1 terminating services to/from penrith &amp; richmond notes\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'trigram\')',
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
        "serialized_db": 'Table Description: Mukaradeeb\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "mukaradeeb \u0645\u0642\u0631 \u0627\u0644\u062f\u064a\u0628" TEXT,\n  "country" TEXT,\n  "province" TEXT,\n  "district" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index mukaradeeb \u0645\u0642\u0631 \u0627\u0644\u062f\u064a\u0628 country province district\n     0 mukaradeeb \u0645\u0642\u0631 \u0627\u0644\u062f\u064a\u0628    iraq al-anbar al-qa\'im\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w0.country ( iraq )\nw0.province ( al-anbar )\nw0.district ( al-qa'im )\ndocuments.title ( mukaradeeb )",
        "question": "Mukaradeeb('Wolf's Den') is a city in Iraq near the Syrian border, in the district of Al-Qa'im, province of Al-Anbar.",
        "blendsql": """
        SELECT (
            {{
                LLMValidate(
                    'Is Mukaradeeb near the Syrian border?', 
                    (SELECT * FROM documents)
                )
            }}
        ) AND EXISTS (
            SELECT * FROM w0 WHERE "district"  = 'al-qa''im' AND "province" = 'al-anbar'
        ) 
        """,
    },
    {
        "serialized_db": 'Table Description: 2006\u201307 San Jose Sharks season\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "no." INTEGER,\n  "cr" INTEGER,\n  "filledcolumnname" TEXT,\n  "gp" INTEGER,\n  "w" INTEGER,\n  "l" INTEGER,\n  "otl" INTEGER,\n  "gf" INTEGER,\n  "ga" INTEGER,\n  "pts" INTEGER\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index  no.  cr filledcolumnname  gp  w  l  otl  gf  ga  pts\n     0    1   2    anaheim ducks  82 48 20   14 258 208  110\n     1    2   5  san jose sharks  82 51 26    5 258 199  107\n     2    3   6     dallas stars  82 50 25    7 226 197  107\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w0.filledcolumnname ( san jose sharks )",
        "question": "The 2006-07 San Jose Sharks season, the 14th season of operation (13th season of play) for the National Hockey League (NHL) franchise, scored the most points in the Pacific Division.",
        "blendsql": """
        SELECT (
            {{
                LLMValidate(
                    'Is the Sharks 2006-07 season the 14th season (13th season of play)?', 
                    (SELECT * FROM documents)
                )
            }}
        ) AND (
            SELECT (SELECT filledcolumnname FROM w0 ORDER BY pts DESC LIMIT 1) = 'san jose sharks'
        )
        """,
    },
    {
        "serialized_db": 'Table Description: Saunders College of Business\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "motto" TEXT,\n  "type" TEXT,\n  "established" INTEGER,\n  "parent institution" TEXT,\n  "dean" TEXT,\n  "academic staff" INTEGER,\n  "students" TEXT,\n  "postgraduates" INTEGER,\n  "location" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index                              motto    type  established                parent institution                  dean  academic staff students  postgraduates                           location\n     0 business and technology - unlocked private         1910 rochester institute of technology jacqueline r. mozrall              30    2400+            346 rochester, new york, united states\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w0.parent institution ( rochester institute of technology )\nw0.dean ( jacqueline r. mozrall )\ndocuments.title ( saunders college of business )",
        "question": "Saunders College of Business, which is accredited by the Association to Advance Collegiate Schools of Business International, is one of the colleges of Rochester Institute of Technology established in 1910 and is currently under the supervision of Dean Jacqueline R. Mozrall.",
        "blendsql": """
        SELECT EXISTS(
            SELECT * FROM w0 
            WHERE "parent institution" = 'rochester institute of technology'
            AND "established" = '1910'
            AND "dean" = 'jacqueline r. mozrall'
        ) AND (
            {{
                LLMValidate(
                    'Is Saunders College of Business (SCB) accredited by the Association to Advance Collegiate Schools of Business International (AACSB)?',
                    (SELECT * FROM documents)
                )
            }}
        )
        """,
    },
    {
        "serialized_db": 'Table Description: 2018 Mexican general election\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "candidate" TEXT,\n  "candidate_2" TEXT,\n  "party" TEXT,\n  "alliance" TEXT,\n  "votes" INTEGER,\n  "%" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index candidate            candidate_2                             party                alliance  votes     %\n     0 candidate    mauricio vila dosal             national action party  por yucata\u0301n al frente 447753  39.6\n     1 candidate mauricio sahui\u0301 rivero institutional revolutionary party      todos por yucata\u0301n 407802 36.09\n     2 candidate    joaqui\u0301n di\u0301az mena    national regeneration movement juntos haremos historia 231330 20.46\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w0.candidate_2 ( mauricio vila dosal )\nw0.party ( national action party )\ndocuments.title ( 2018 mexican general election )",
        "question": "Mauricio Vila Dosal of the National Action Party overwhelmingly won the race for Governor of Yucatán during the 2018 Mexican general election.",
        "blendsql": """
        SELECT (
            {{
                LLMValidate(
                    'Did Mauricio Vila Dosal of the National Action Party overwhelmingly win the race for Governor of Yucatán during the 2018 Mexican general election?',
                    (SELECT * FROM w0)
                ) 
            }}
        )
        """,
    },
    {
        "serialized_db": 'Table Description: Johannes Voggenhuber\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "attribute" TEXT,\n  "value" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index                    attribute                        value\n     0         johannes voggenhuber         johannes voggenhuber\n     1 spokesman of the green party spokesman of the green party\n     2                  preceded by          freda meissner-blau\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w0.attribute ( spokesman of the green party , johannes voggenhuber )\nw0.value ( spokesman of the green party , johannes voggenhuber )\ndocuments.title ( johannes voggenhuber )",
        "question": "Johannes Voggenhuber(born  5 June 1950) a politician who served as the former spokesman of the Green Party from Austria.",
        "blendsql": """
        SELECT (
            {{
                LLMValidate(
                    'Was Johannes Voggenhuber born on 5 June 1950?',
                    (SELECT * FROM documents)
                ) 
            }}
        ) AND (
            {{
                LLMValidate(
                    'Was Johannes Voggenhuber former Member of the European Parliament (MEP) for the Austrian Green Party, which is part of the European Greens?',
                    (SELECT * FROM documents)
                ) 
            }}
        )
        """,
    },
    {
        "serialized_db": 'Table Description: Jan Moir\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "jan moir" TEXT,\n  "born" TEXT,\n  "nationality" TEXT,\n  "occupation" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index jan moir            born nationality                     occupation\n     0 jan moir 1958-8 (age 62)     british columnist, restaurant reviewer\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w0.jan moir ( jan moir )\ndocuments.title ( journalist , jan moir )",
        "question": "Sixty two year old Welsh journalist Jan Moir worked for a couple other papers before working at Daily Mail as an opinion columnist and has won several awards for her writing.",
        "blendsql": """
        SELECT (
            SELECT {{LLMMap('What age?', 'w0::born')}} = 62 FROM w0
        ) AND (
            {{
                LLMValidate(
                    'Did Jan Moir work at a couple other papers before working at Daily Mail as an opinion columnist?',
                    (SELECT * FROM documents)
                ) 
            }}
        ) AND (
            {{
                LLMValidate(
                    'Has Jan Moir won several awards for her writing?',
                    (SELECT * FROM documents)
                ) 
            }}
        )
        """,
    },
    {
        "serialized_db": 'Table Description: Paspels\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "languages in paspels" TEXT,\n  "languages in paspels_2" TEXT,\n  "languages in paspels_3" TEXT,\n  "languages in paspels_4" TEXT,\n  "languages in paspels_5" TEXT,\n  "languages in paspels_6" TEXT,\n  "languages in paspels_7" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index languages in paspels languages in paspels_2 languages in paspels_3 languages in paspels_4 languages in paspels_5 languages in paspels_6 languages in paspels_7\n     0            languages            census 1980            census 1980            census 1990            census 1990            census 2000            census 2000\n     1            languages                 number                percent                 number                percent                 number                percent\n     2               german                    246                 77.36%                    320                 89.39%                    386                 94.38%\n*/',
        "bridge_hints": "w0.languages in paspels ( romanish , languages )",
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
        "serialized_db": 'Table Description: Adrian Barbullushi\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "personal information" TEXT,\n  "personal information_2" TEXT,\n  "personal information_3" TEXT,\n  "personal information_4" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index personal information personal information_2 personal information_3 personal information_4\n     0        date of birth   personal information   personal information   personal information\n     1       place of birth                albania                albania                albania\n     2  playing position(s)             midfielder             midfielder             midfielder\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w0.personal information_2 ( ionikos , egaleo )\nw0.personal information_3 ( albania )\nw0.personal information_4 ( albania )\ndocuments.title ( adrian barbullushi )",
        "question": "Retired Albanian football player Adrian Barbullushi never played for Egaleo or Ionikos.",
        "blendsql": """
        SELECT NOT EXISTS(
            SELECT * FROM w0 WHERE "personal information_2" = 'egaleo'
        ) AND NOT EXISTS (
            SELECT * FROM w0 WHERE "personal information_2" = 'ionikos'
        )
        """,
    },
    {
        "serialized_db": 'Table Description: 1994 Temple Owls football team\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "date" TEXT,\n  "time" TEXT,\n  "opponent" TEXT,\n  "site" TEXT,\n  "result" TEXT,\n  "attendance" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index         date  time       opponent site  result attendance\n     0  september 3  time      at akron* site  w 32\u20137 attendance\n     1 september 17 t18:0 east carolina* site l 14\u201331       9137\n     2 september 24 t18:0       at army* site w 23\u201320       9137\n*/',
        "bridge_hints": "w0.date ( october 22 , november 5 )",
        "question": "As part of their schedule, the Temple Owls football team played at Miami on November 5, 1994, losing 21–38, and played at Syracuse on October 22, losing 42–49.",
        "blendsql": """
        SELECT EXISTS(
            SELECT * FROM w0 WHERE date = 'november 5' AND {{LLMMap('Is this in Miami?', 'w0::opponent')}} = TRUE AND {{LLMMap('Did they lose 21-38?', 'w0::result')}} = TRUE
        ) AND EXISTS(
            SELECT * FROM w0 WHERE date = 'october 22' AND {{LLMMap('Is this Syracuse?', 'w0::opponent')}} = TRUE AND {{LLMMap('Did they lose 42-49?', 'w0::result')}} = TRUE
        )
        """,
    },
    {
        "serialized_db": 'Table Description: Leon Haslam\nCREATE TABLE "w0" (\n"index" INTEGER,\n  "season" INTEGER,\n  "series" TEXT,\n  "motorcycle" TEXT,\n  "team" TEXT,\n  "race" INTEGER,\n  "win" INTEGER,\n  "podium" INTEGER,\n  "pole" INTEGER,\n  "flap" INTEGER,\n  "pts" INTEGER,\n  "plcd" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w0" LIMIT 3\n index  season series   motorcycle          team  race  win  podium  pole  flap  pts plcd\n     0    1998  125cc honda rs125r honda britain     1    0       0     0     0    0   nc\n     1    1999  125cc honda rs125r honda britain     1    0       0     0     0    0   nc\n     2    2000  125cc italjet f125  italjet moto    15    0       0     0     0    6 27th\n*/',
        "bridge_hints": "w0.series ( british superbike , superbike )",
        "question": "Leon Haslam raced in the British Superbike Championship four years in a row, from 2005-2008, placing second in both 2006 and 2008.",
        "blendsql": """
        SELECT (SELECT COUNT(DISTINCT season) = 4 FROM w0 WHERE series = 'british superbike' AND season BETWEEN 2005 AND 2008)
        AND (SELECT plcd = '2nd' FROM w0 WHERE series = 'british superbike' AND season = 2006)
        AND (SELECT plcd = '2nd' FROM w0 WHERE series = 'british superbike' AND season = 2008)
        """,
    },
]

sql_examples = []
