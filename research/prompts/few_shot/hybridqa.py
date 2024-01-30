# {
#     "serialized_db": "",
#     "bridge_hints": "",
#     "question": "",
#     "blendsql": """
#        """
# }
blendsql_examples = [
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "name" TEXT,\n  "province" TEXT,\n  "city" TEXT,\n  "year" TEXT,\n  "remarks" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index                      name          province     city year                                                         remarks\n     0       abdul rahman mosque    kabul province    kabul 2009                                   largest mosque in afghanistan\n     1 friday mosque of kandahar kandahar province kandahar 1750                houses the cloak of the islamic prophet muhammad\n     2     omar al-farooq mosque kandahar province kandahar 2014 built on the site that was a popular cinema of kandahar . [ 1 ]\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w.city ( herat )\ndocuments.title ( herat , fire temple )",
        "question": "Who were the builders of the mosque in Herat with fire temples ?",
        "blendsql": """
        {{
            LLMQA(
                'Who were the builders of the mosque?',
                (
                    SELECT documents.title AS 'Building', documents.content FROM documents
                    JOIN {{
                        LLMJoin(
                            left_on='w::name',
                            right_on='documents::title'
                        )
                    }}
                    WHERE w.city = 'herat' AND w.remarks LIKE '%fire temple%'
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "no" INTEGER,\n  "rider" TEXT,\n  "team" TEXT,\n  "motorcycle" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  no          rider                 team      motorcycle\n     0   1   carl fogarty   ducati performance      ducati 996\n     1   4 akira yanagawa kawasaki racing team kawasaki zx-7rr\n     2   5  colin edwards        castrol honda      honda rc45\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "",
        "question": "After what season did the number 7 competitor retire ?",
        "blendsql": """
        {{
            LLMQA(
                'When did the competitor retire?',
                (
                    SELECT documents.title AS 'Competitor', documents.content FROM documents
                    JOIN {{
                        LLMJoin(
                            left_on='w::rider',
                            right_on='documents::title'
                        )
                    }}
                    WHERE w.no = 7
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "year" TEXT,\n  "winner" TEXT,\n  "position" TEXT,\n  "school" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index    year         winner   position     school\n     0 1961-62       ron ryan right wing      colby\n     1 1962-63 bob brinkworth     center rensselaer\n     2 1963-64 bob brinkworth     center rensselaer\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w.year ( 1971-72 )",
        "question": "What year was the 1971-72 ECAC Hockey Player of the Year born ?",
        "blendsql": """
        {{
            LLMQA(
                'What year was the player born?',
                (
                    SELECT documents.title AS 'Player', documents.content FROM documents
                    JOIN {{
                        LLMJoin(
                            left_on = 'w::winner',
                            right_on = 'documents::title'
                        )
                    }}
                    WHERE w.year = '1971-72'
                )
            )
        }}
        """,
    },
    # {
    #     "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "season" TEXT,\n  "team" TEXT,\n  "record" TEXT,\n  "playoffs result" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index season              team          record            playoffs result\n     0 2004-5 san antonio spurs  59-23 ( 0.72 )             won nba finals\n     1 2005-6 san antonio spurs 63-19 ( 0.768 ) lost conference semifinals\n     2 2006-7  dallas mavericks 67-15 ( 0.817 )           lost first round\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
    #     "bridge_hints": "",
    #     "question": "What season was it that this team was hoping to improve upon their 60-22 output from the previous season ?",
    #     "blendsql": """
    #     {{
    #         LLMQA(
    #             'In which season was this team hoping to improve upon their 60-22 output?',
    #             (SELECT title, content FROM documents WHERE content LIKE '%60-22%')
    #         )
    #     }}
    #     """,
    # },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "name" TEXT,\n  "unit" TEXT,\n  "date of action" TEXT,\n  "place of action" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index              name                     unit          date of action                                 place of action\n     0     william allan 24 24th regiment of foot  1879-2-22 22-1879-1-23          battle of rorke s drift , natal colony\n     1 william beresford            9 9th lancers    1879-7-3 3 july 1879 white umfolozi river ( near ulundi ) , zululand\n     2     anthony booth 80 80th regiment of foot 1879-3-12 12 march 1879                   battle of intombe , transvaal\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "question": "What battle did the man born on 7 December 1839 fight in ?",
        "bridge_hints": "",
        "blendsql": """
        SELECT {{LLMMap('Name of the battle?', 'w::place of action')}} FROM w WHERE name = {{
            LLMQA(
                'Who was born on 7 December 1839?',
                (
                    SELECT documents.title, documents.content FROM documents JOIN {{
                        LLMJoin(
                            left_on='w::name',
                            right_on='documents::title'
                        )
                    }}
                )
                options = 'w::name'
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "name" TEXT,\n  "location" TEXT,\n  "type" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index            name        location                              type\n     0    allianz park hendon , london rugby union and athletics stadium\n     1        kia oval      kennington                   cricket stadium\n     2 banks s stadium         walsall                  football stadium\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w.name ( kia oval )",
        "question": "What is the borough in which Kia Oval is located ?",
        "blendsql": """
        {{
            LLMQA(
                'What borough is the Kia Oval located in?',
                (
                    SELECT documents.title, documents.content FROM documents 
                    JOIN {{
                        LLMJoin(
                            left_on='w::name',
                            right_on='documents::title'
                        )
                    }} WHERE w.name = 'kia oval'
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "venue" TEXT,\n  "sports" TEXT,\n  "capacity" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index                venue                                                                     sports   capacity\n     0           bjela\u0161nica                                                      alpine skiing ( men ) not listed\n     1   igman , malo polje                              nordic combined ( ski jumping ) , ski jumping not listed\n     2 igman , veliko polje biathlon , cross-country skiing , nordic combined ( cross-country skiing ) not listed\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "",
        "question": "What is the capacity of the venue that was named in honor of Juan Antonio Samaranch in 2010 after his death ?",
        "blendsql": """
        SELECT capacity FROM w WHERE venue = {{
            LLMQA(
                'Which venue was named in honor of Juan Antonio Samaranch after his death?',
                (SELECT title AS 'Venue', content FROM documents WHERE documents MATCH 'juan OR antonio OR samaranch')
                
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "round ( pick )" TEXT,\n  "name" TEXT,\n  "position" TEXT,\n  "school" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index round ( pick )           name             position                 school\n     0       1 ( 20 )  joshua fields right-handed pitcher  university of georgia\n     1       2 ( 20 )   dennis raben           outfielder    university of miami\n     2       3 ( 98 ) aaron pribanic right-handed pitcher university of nebraska\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w.school ( university of georgia )\ndocuments.title ( university of georgia )",
        "question": "Which teams has the player drafted by the Seattle Mariners in 2008 out of University of Georgia played for in the MLB ?",
        "blendsql": """
        {{
            LLMQA(
                'Which teams drafted this player?',
                (
                    SELECT documents.title AS 'Player', documents.content FROM documents
                    JOIN {{
                        LLMJoin(
                            left_on='w::name',
                            right_on='documents::title'
                        )
                    }}
                    WHERE w.school = 'university of georgia'
                )
            )
        }}
       """,
    },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "rank" INTEGER,\n  "film" TEXT,\n  "year" INTEGER,\n  "director" TEXT,\n  "studio ( s )" TEXT,\n  "worldwide gross" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  rank                  film  year             director                                       studio ( s )                worldwide gross\n     0     1       amazon obhijaan  2017 kamaleswar mukherjee                              shree venkatesh films \u20b9 48.63 crore ( us $ 6800000 )\n     1     2         chander pahar  2013 kamaleswar mukherjee                              shree venkatesh films    \u20b9 15 crore ( us $ 2100000 )\n     2     3 boss 2 : back to rule  2017           baba yadav jeetz fireworks walzen media works jaaz multimedia  \u20b9 10.5 crore ( us $ 1500000 )\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "",
        "question": "The story of a cab driver witnessing a murder by a criminal kingpin leads to extensive loss in an Indian film directed by one of the leading ad film makers in Kolkata who has made how many ad films in his career ?",
        "blendsql": """
        {{
            LLMQA(
                'How many ad films has this director made in their carer?',
                (
                    SELECT title AS 'Director', content FROM documents
                    JOIN {{
                        LLMJoin(
                            left_on='w::director',
                            right_on='documents::title'
                        )
                    }} WHERE w.film = {{
                        LLMQA(
                            'Which film is about a cab driver witnessing a murder by a criminal kingpin?',
                            (SELECT title, content FROM documents WHERE documents MATCH 'cab OR murder OR kingpin'),
                            options='w::film'
                        )
                    }}
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "camp name" TEXT,\n  "council" TEXT,\n  "location" TEXT,\n  "status" TEXT,\n  "notes" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index                             camp name                    council                       location status                                                                                                                                                     notes\n     0 camp aquila ( formerly camp mauwehu ) connecticut yankee council candlewood lake , sherman , ct closed                                                  located on candlewood lake in sherman , ct. , the camp was sold in 1982 along with camp toquam in goshen\n     1                      camp cochipianee       bristol area council                    goshen , ct closed the camp was founded in 1928 by the bristol area council and was sold after the new britain area council and the bristol area council were merged in 1972\n     2                           camp irving         housatonic council                   shelton , ct closed             the camp was located in shelton in the birchbank area along the housatonic river . it was closed in 1945 and the buildings were razed in 1948\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "",
        "question": "What is the status of the camp in the town that split from Stonington in 1724 ?",
        "blendsql": """
        SELECT status FROM w WHERE location = {{
            LLMQA(
                'Which town split from Stonington in 1724?',
                (
                    SELECT documents.title, documents.content FROM documents 
                        WHERE documents MATCH 'stonington'
                ),
                options='w::location'
            )
        }}
        """,
    },
    # {
    #     "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "filledcolumnname" INTEGER,\n  "name on the register" TEXT,\n  "date listed" TEXT,\n  "location" TEXT,\n  "city or town" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  filledcolumnname    name on the register               date listed                                                                                                                                                             location city or town\n     0                 1 carnegie public library  1986-7-24 ( # 86001934 )                                                 447 4th ave. 48\u00b032\u203257\u2033n 109\u00b040\u203235\u2033w / 48.549167\u00b0n 109.676389\u00b0w / 48.549167 ; -109.676389 ( carnegie public library )        havre\n     1                 2     h. earl clack house 1985-10-24 ( # 85003385 )                                                             532 2nd ave. 48\u00b032\u203254\u2033n 109\u00b040\u203248\u2033w / 48.548333\u00b0n 109.68\u00b0w / 48.548333 ; -109.68 ( h. earl clack house )        havre\n     2                 3       fort assinniboine  1989-5-31 ( # 89000040 ) county route 82nd ave. west , 0.5 miles southeast of u.s. route 87 48\u00b029\u203259\u2033n 109\u00b047\u203239\u2033w / 48.499722\u00b0n 109.794167\u00b0w / 48.499722 ; -109.794167 ( fort assinniboine )        havre\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
    #     "bridge_hints": "",
    #     "question": "What year was the building built with date listed # 94000865 ?",
    #     "blendsql": """
    #     {{
    #         LLMQA(
    #             'What year was it built?',
    #             (
    #                 SELECT documents.title AS 'Building', documents.content FROM documents
    #                     JOIN {{
    #                         LLMJoin(
    #                             left_on='w::name on the register',
    #                             right_on='documents::title'
    #                         )
    #                     }} WHERE w."date listed" LIKE '%# 94000865%'
    #             )
    #         )
    #     }}
    #     """,
    # },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "medal" TEXT,\n  "name" TEXT,\n  "sport" TEXT,\n  "event" TEXT,\n  "time/score" TEXT,\n  "date" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "w" LIMIT 3\n index medal                                     name               sport                   event time/score    date\n     0  gold andrew gemmell sean ryan ashley twichell open water swimming         5 km team event     57:0.6 july 21\n     1  gold                             dana vollmer            swimming women s 100 m butterfly      56.87 july 25\n     2  gold                              ryan lochte            swimming   men s 200 m freestyle    1:44.44 july 26\n*/\n\nCREATE VIRTUAL TABLE documents USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "",
        "bridge_hints": "",
        "question": "What is the name of the oldest person whose result , not including team race , was above 2 minutes ?",
        "blendsql": """
        {{
            LLMQA(
                'Of the athletes here, who is older?',
                (
                    SELECT DISTINCT documents.title AS 'Athlete', documents.content FROM documents
                    JOIN {{
                        LLMJoin(
                            left_on='w::name',
                            right_on='documents::title'
                        )
                    }} WHERE {{
                        LLMMap('Is this time greater than 2 minutes?', 'w::time/score')
                    }} = TRUE AND {{
                        LLMMap('Is this a team race?', 'w::event')
                    }} = FALSE
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "year" INTEGER,\n  "award" TEXT,\n  "category" TEXT,\n  "nominated work" TEXT,\n  "result" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  year                                      award                                                category    nominated work    result\n     0  2013           critics  choice television award             best supporting actor in a movie/miniseries political animals nominated\n     1  2013                           gold derby award                          tv movie/mini supporting actor political animals nominated\n     2  2013 online film & television association award best supporting actor in a motion picture or miniseries political animals nominated\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "",
        "question": "How many social media sites are used to gather votes for the 2016 award ?",
        "blendsql": """
        {{
            LLMQA(
                'How many social media sites are used to gather votes?',
                (
                    select documents.title, documents.content from documents
                        JOIN {{
                            LLMJoin(
                                left_on='w::award',
                                right_on='documents::title'
                            )
                        }} WHERE w.year = 2016
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "w" (\n"index" INTEGER,\n  "date" TEXT,\n  "language" TEXT,\n  "language family" TEXT,\n  "region" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index                     date language language family      region\n     0 early 2nd millennium bce sumerian         isolate mesopotamia\n     1       2nd millennium bce  eblaite         semitic       syria\n     2            ca . 1100 bce  hittite       anatolian    anatolia\n*/\n\nCREATE VIRTUAL TABLE "documents" USING fts5(title, content, tokenize = \'trigram\')',
        "bridge_hints": "w.region ( mesopotamia )\ndocuments.title ( mesopotamia )",
        "question": "What was the language family that was used in Hattusa , as well as parts of the northern Levant and Upper Mesopotamia ?",
        "blendsql": """
        SELECT "language family" FROM w 
        WHERE language = {{
            LLMQA(
                'Which language was used in Hattusa, as well as parts of the northern Levant and Upper Mesopotamia ?',
                (SELECT title, content FROM documents WHERE documents MATCH 'hattusa'),
                options='w::language'
            )
        }}
       """,
    },
]

sql_examples = []
