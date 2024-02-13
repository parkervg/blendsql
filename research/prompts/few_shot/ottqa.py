blendsql_examples = [
    {
        "serialized_db": 'CREATE TABLE "./List of Rangers F.C. records and statistics (0)" (\n"#" INTEGER,\n  "player" TEXT,\n  "to" TEXT,\n  "fee" TEXT,\n  "date" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of Rangers F.C. records and statistics (0)" LIMIT 3\n #                   player                to      fee      date\n 1              alan hutton tottenham hotspur \u00a39000000 2008-1-30\n 2 giovanni van bronckhorst           arsenal \u00a38500000 2001-6-20\n 3      jean-alain boumsong  newcastle united \u00a38000000  2005-1-1\n*/\n\nCREATE TABLE "./List of Rangers F.C. records and statistics (1)" (\n"#" INTEGER,\n  "player" TEXT,\n  "from" TEXT,\n  "fee" TEXT,\n  "date" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of Rangers F.C. records and statistics (1)" LIMIT 3\n #         player      from       fee       date\n 1 tore andr\u00e9 flo   chelsea \u00a312000000 2000-11-23\n 2      ryan kent liverpool  \u00a36500000   2019-9-2\n 2   michael ball   everton  \u00a36500000  2001-8-20\n*/\n\nCREATE TABLE "./List of Rangers F.C. players (2)" (\n"inductee" TEXT,\n  "induction year" TEXT,\n  "position" TEXT,\n  "rangers career" TEXT,\n  "appearances" INTEGER,\n  "honours" TEXT,\n  "interntional caps" INTEGER\n)\n/*\n3 example rows:\nSELECT * FROM "./List of Rangers F.C. players (2)" LIMIT 3\n      inductee induction year position rangers career  appearances honours  interntional caps\n  moses mcneil           2000       mf      1872-1882           34    none                  2\n  peter mcneil           2010       mf      1872-1877            7    none                  0\npeter campbell           2010       fw      1872-1879           24    none                  2\n*/',
        "bridge_hints": "",
        "question": "When was the third highest paid Rangers F.C . player born ?",
        "blendsql": """
        {{
            LLMQA(
                'When was the Rangers Player born?'
                (
                    WITH t AS (
                        SELECT player FROM (
                            SELECT * FROM "./List of Rangers F.C. records and statistics (0)"
                            UNION ALL SELECT * FROM "./List of Rangers F.C. records and statistics (1)"
                        ) ORDER BY trim(fee, '£') DESC LIMIT 1 OFFSET 2
                    ), d AS (
                        SELECT * FROM documents JOIN t WHERE documents MATCH '"' || t.player || '"' || ' OR rangers OR fc' ORDER BY rank LIMIT 5
                    ) SELECT d.content, t.player AS 'Rangers Player' FROM d JOIN t
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "./2006 League of Ireland Premier Division (1)" (\n"team" TEXT,\n  "manager" TEXT,\n  "main sponsor" TEXT,\n  "kit supplier" TEXT,\n  "stadium" TEXT,\n  "capacity" INTEGER\n)\n/*\n3 example rows:\nSELECT * FROM "./2006 League of Ireland Premier Division (1)" LIMIT 3\n          team           manager      main sponsor kit supplier          stadium  capacity\n     bohemians   gareth farrelly des kelly carpets     o\'neills   dalymount park      8500\nbray wanderers     eddie gormley      slevin group       adidas carlisle grounds      7000\n     cork city damien richardson            nissan     o\'neills    turners cross      8000\n*/\n\nCREATE TABLE "./2006 League of Ireland Premier Division (5)" (\n"team" TEXT,\n  "manager" TEXT,\n  "main sponsor" TEXT,\n  "kit supplier" TEXT,\n  "stadium" TEXT,\n  "capacity" INTEGER\n)\n/*\n3 example rows:\nSELECT * FROM "./2006 League of Ireland Premier Division (5)" LIMIT 3\n          team           manager      main sponsor kit supplier          stadium  capacity\n     bohemians   gareth farrelly des kelly carpets     o\'neills   dalymount park      8500\nbray wanderers     eddie gormley      slevin group       adidas carlisle grounds      7000\n     cork city damien richardson            nissan     o\'neills    turners cross      8000\n*/\n\nCREATE TABLE "./2006 SK Brann season (2)" (\n"date" TEXT,\n  "host" TEXT,\n  "agg" TEXT,\n  "visitor" TEXT,\n  "ground" TEXT,\n  "attendance" TEXT,\n  "tournament" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./2006 SK Brann season (2)" LIMIT 3\n   date      host agg     visitor                     ground attendance                tournament\n 2 july   ham-kam 4-0       brann briskeby gressbane , hamar       6218               tippeligaen\n 5 july     brann 3-1 levanger il     brann stadion , bergen       1948             norwegian cup\n13 july glentoran 0-1       brann         the oval , belfast       1743 uefa cup qualifying round\n*/',
        "bridge_hints": "",
        "question": "The home stadium of the Bray Wanderers of 2006 League of Ireland is situated behind what station ?",
        "blendsql": """
        {{  
            LLMQA(
                'What station is the Bray Wanderers home stadium situated behind?',
                (
                    WITH t AS (
                        SELECT stadium FROM "./2006 League of Ireland Premier Division (1)" WHERE team = 'bray wanderers'
                    ), d AS (
                        SELECT * FROM documents JOIN t WHERE documents MATCH '"' || t.stadium || '"' ORDER BY rank LIMIT 5
                    ) SELECT d.content, t.stadium AS 'Home Stadium' FROM d JOIN t
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "./List of medical recipients of the Victoria Cross (0)" (\n"name" TEXT,\n  "regiment/corps" TEXT,\n  "date" TEXT,\n  "conflict" TEXT,\n  "location" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of medical recipients of the Victoria Cross (0)" LIMIT 3\n          name           regiment/corps                         date            conflict      location\nharold ackroyd royal berkshire regiment (xxxx-7-311917-8-1,p-38715d)     first world war passchendaele\n william allen          royal artillery                     1916-9-3     first world war   near mesnil\n henry andrews  indian medical services                   1919-10-22 waziristan campaign  khajuri post\n*/\n\nCREATE TABLE "./List of living recipients of the George Cross (0)" (\n"name" TEXT,\n  "year of award" INTEGER,\n  "location of gallantry" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of living recipients of the George Cross (0)" LIMIT 3\n                       name  year of award                 location of gallantry\n             henry flintoff           1944            farndale , north yorkshire\n                   alf lowe           1949             portland harbour , dorset\nmargaret purves nee vaughan           1949 near sully island , vale of glamorgan\n*/\n\nCREATE TABLE "./List of Australian Victoria Cross recipients (0)" (\n"name" TEXT,\n  "date of action" TEXT,\n  "conflict" TEXT,\n  "unit" TEXT,\n  "place of action" TEXT,\n  "location of medal" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of Australian Victoria Cross recipients (0)" LIMIT 3\n            name date of action         conflict                          unit     place of action location of medal\ncharles anderson           1942 second world war              2/19th battalion muar river , malaya               awm\n   thomas axford           1918  first world war                16th battalion hamel wood , france               awm\n    peter badcoe          1967*      vietnam war australian army training team huong tra , vietnam               awm\n*/',
        "bridge_hints": "",
        "question": "What nationality was the winner of the 1945 award of the Victoria Cross ?",
        "blendsql": """
        {{
            LLMQA(
                "What was the Victoria Cross recipient's nationality?",
                (
                    WITH t AS (
                        SELECT name FROM "./List of medical recipients of the Victoria Cross (0)"
                            WHERE SUBSTR(date, 0, 5) = '1945'
                    ), d AS (
                        SELECT * FROM documents JOIN t WHERE documents MATCH '"' || t.name || '"' || ' OR victoria cross' ORDER BY rank LIMIT 5
                    ) SELECT d.content, t.name AS recipient FROM d JOIN t
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "./Atlantic Hockey Player of the Year (1)" (\n"year" TEXT,\n  "winner" TEXT,\n  "position" TEXT,\n  "school" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./Atlantic Hockey Player of the Year (1)" LIMIT 3\n   year           winner   position       school\n2019-20     jason cotton    forward sacred heart\n2018-19    joseph duszak defenceman   mercyhurst\n2017-18 dylan mclaughlin    forward      cansius\n*/\n\nCREATE TABLE "./List of Atlantic Hockey Most Valuable Player in Tournament (0)" (\n"year" INTEGER,\n  "winner" TEXT,\n  "position" TEXT,\n  "school" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of Atlantic Hockey Most Valuable Player in Tournament (0)" LIMIT 3\n year          winner  position     school\n 2004     greg kealey   forward holy cross\n 2005 scott champagne left wing mercyhurst\n 2006  james sixsmith left wing holy cross\n*/\n\nCREATE TABLE "./Atlantic Hockey Player of the Year (0)" (\n"year" TEXT,\n  "winner" TEXT,\n  "position" TEXT,\n  "school" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./Atlantic Hockey Player of the Year (0)" LIMIT 3\n   year           winner   position     school\n2018-19    joseph duszak defenceman mercyhurst\n2017-18 dylan mclaughlin    forward    cansius\n2016-17 charles williams goaltender    cansius\n*/',
        "bridge_hints": "",
        "question": "Which NHL team has the Player of the Year of Atlantic Hockey for the season ending in 2019 signed a agreement with ?",
        "blendsql": """
        {{
            LLMQA(
                'Which team has the NHL player signed with?',
                (
                    WITH t AS (
                        SELECT winner FROM (
                            SELECT * FROM "./Atlantic Hockey Player of the Year (0)"
                            UNION ALL SELECT * FROM "./Atlantic Hockey Player of the Year (1)"
                        ) AS w WHERE {{LLMMap('Does this end in 2019?', 'w::year')}} = TRUE
                    ), d AS (
                        SELECT * FROM documents JOIN t WHERE documents MATCH '"' || t.winner || '"' || ' OR hockey' ORDER BY rank LIMIT 5
                    ) SELECT d.content, t.winner AS 'NHL Player' FROM d JOIN t
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "./Cuba at the UCI Track Cycling World Championships (1)" (\n"name" TEXT,\n  "event" TEXT,\n  "result" TEXT,\n  "rank" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./Cuba at the UCI Track Cycling World Championships (1)" LIMIT 3\n           name                       event         result rank\nlisandra guerra             women \'s sprint 11.121 ( q ) ,   18\nlisandra guerra   women \'s 500 m time trial         34.226    9\n marlies mejias women \'s individual pursuit        3:35.57    8\n*/\n\nCREATE TABLE "./Cuba at the UCI Track Cycling World Championships (2)" (\n"medal" TEXT,\n  "championship" TEXT,\n  "name" TEXT,\n  "event" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./Cuba at the UCI Track Cycling World Championships (2)" LIMIT 3\n medal   championship            name                     event\n  gold 2003 stuttgart yoanka gonz\u00e1lez      women \'s points race\n  gold 2004 melbourne yoanka gonz\u00e1lez          women \'s scratch\nbronze  2006 bordeaux lisandra guerra women \'s 500 m time trial\n*/\n\nCREATE TABLE "./Cuba at the UCI Track Cycling World Championships (0)" (\n"name" TEXT,\n  "event" TEXT,\n  "result" TEXT,\n  "rank" INTEGER\n)\n/*\n3 example rows:\nSELECT * FROM "./Cuba at the UCI Track Cycling World Championships (0)" LIMIT 3\n                     name                     event    result  rank\nlisandra guerra rodriguez           women \'s sprint      none    30\nlisandra guerra rodriguez women \'s 500 m time trial pt34.692s     9\nlisandra guerra rodriguez           women \'s keirin      none    13\n*/',
        "bridge_hints": "",
        "question": "In which Track Cycling World Championships event was the person born in Matanzas , Cuba ranked highest ?",
        "blendsql": """
        {{
            LLMQA(
                'In what event was the cyclist ranked highest?',
                (
                    SELECT * FROM (
                        SELECT * FROM "./Cuba at the UCI Track Cycling World Championships (2)"
                    ) as w WHERE w.name = {{
                        LLMQA(
                            "Which cyclist was born in Matanzas, Cuba?",
                            (
                                SELECT * FROM documents 
                                    WHERE documents MATCH 'matanzas OR cycling OR track OR born' 
                                    ORDER BY rank LIMIT 3
                            ),
                            options="w::name"
                        )
                    }}
                ),
                options='w::event'
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "./2011 Thai Premier League (1)" (\n"team" TEXT,\n  "sponsor" TEXT,\n  "kit maker" TEXT,\n  "team captain" TEXT,\n  "head coach" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./2011 Thai Premier League (1)" LIMIT 3\n           team      sponsor kit maker      team captain           head coach\n    army united        chang       pan  wanchana rattana       adul rungruang\n  bangkok glass     leo beer     umbro    amnaj kaewkiew arjhan srong-ngamsub\nbec tero sasana 3000 battery       fbt teeratep winothai     phayong khunnaen\n*/\n\nCREATE TABLE "./2013 Thai Premier League (5)" (\n"team" TEXT,\n  "head coach" TEXT,\n  "captain" TEXT,\n  "kit manufacturer" TEXT,\n  "shirt sponsor" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./2013 Thai Premier League (5)" LIMIT 3\n          team        head coach           captain kit manufacturer shirt sponsor\n   army united alexandr\u00e9 p\u00f6lking   chaiwat nak-iem              pan         chang\n bangkok glass attaphol buspakom teeratep winothai            umbro      leo beer\nbangkok united  sasom pobprasert nattaporn phanrit              fbt          true\n*/\n\nCREATE TABLE "./2012 Thai Premier League (0)" (\n"team" TEXT,\n  "sponsor" TEXT,\n  "kit maker" TEXT,\n  "team captain" TEXT,\n  "head coach" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./2012 Thai Premier League (0)" LIMIT 3\n           team   sponsor kit maker          team captain          head coach\n    army united     chang       pan        tatree sing-ha    paniphon kerdyam\n  bangkok glass  leo beer     umbro       amnart kaewkiew       phil stubbins\nbec tero sasana channel 3       fbt rangsan viwatchaichok sven-g\u00f6ran eriksson\n*/',
        "bridge_hints": "",
        "question": "What is the home stadium of the team Buriram United whose team captain is Apichet Puttan ?",
        "blendsql": """
        {{
            LLMQA(
                'What is the home stadium of Buriram United?',
                (
                    SELECT * FROM documents WHERE documents MATCH 'buriram united' ORDER BY rank LIMIT 5
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "./List of fictional canines in animation (2)" (\n"name" TEXT,\n  "species" TEXT,\n  "origin" TEXT,\n  "notes" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of fictional canines in animation (2)" LIMIT 3\n name species                origin                                                                                                                                                                                                                                                                                                                                                                                                notes\n aleu wolfdog balto ii : wolf quest                                                                                                                                                                                                                                                                                                                                                              aleu is a wolfdog like her father balto\nbalto wolfdog                 balto balto is a wolf-dog hybrid , shunned by both humans and dogs in the town of nome . he is a rugged spirit , adventurer of his social domain ; a rebel soul , no 1 to turn to but himself . his only friends are boris , a russian goose , jenna , a siberian husky and muk and luk , 2 polar bears . balto and boris live on a grounded boat outside nome , while muk and luk are occasional visitors\ndanny   dingo           blinky bill                                                                                                                                                                                                                                                                                                                                 oldest brother of the family and main antagonist of the first season\n*/\n\nCREATE TABLE "./List of fictional canines in animation (1)" (\n"name" TEXT,\n  "origin" TEXT,\n  "notes" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of fictional canines in animation (1)" LIMIT 3\n                name                       origin                                                               notes\n  antoine d\'coolette           sonic the hedgehog                                                                none\nbent-tail the coyote various walt disney cartoons a brown coyote who appeared as a nemesis of pluto in a few cartoons\n    bent-tail junior various walt disney cartoons                                      bent-tail \'s unintelligent son\n*/\n\nCREATE TABLE "./List of fictional canines in animation (0)" (\n"name" TEXT,\n  "origin" TEXT,\n  "notes" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of fictional canines in animation (0)" LIMIT 3\n       name                   origin                                                                                                                                     notes\nbrother fox        song of the south                                              fox who tries to eat br\'er rabbit and often collaborates with br\'er fox to achieve this goal\nbrother fox                 coonskin a satirical subversion of joel chandler harris and disney \'s similar character from song of the south , reimagined as an african-american\n  cajun fox courage the cowardly dog                                                                                                                                      none\n*/',
        "bridge_hints": "",
        "question": "What is the setting of the animated series featuring the fictional canine Daisy the Dingo ?",
        "blendsql": """
        {{
            LLMQA(
                'Where is the animated TV series set?',
                (
                    WITH t AS (
                        SELECT origin FROM "./List of fictional canines in animation (2)" AS w
                        WHERE w.name = 'daisy' AND w.species = 'dingo'
                    ), d AS (
                        SELECT * FROM documents JOIN t WHERE documents MATCH '"' || t.origin || '"' || ' OR animated OR set' ORDER BY rank LIMIT 5
                    ) SELECT d.content, t.origin AS 'Animated TV Series' FROM d JOIN t 
                )
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "./Primera B Nacional (0)" (\n"season" TEXT,\n  "champion" TEXT,\n  "runner-up" TEXT,\n  "third place" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./Primera B Nacional (0)" LIMIT 3\n season          champion        runner-up    third place\n1986-87 deportivo armenio         banfield       belgrano\n1987-88 deportivo mandiy\u00fa san martin ( t ) chaco for ever\n1988-89    chaco for ever            uni\u00f3n          col\u00f3n\n*/\n\nCREATE TABLE "./Categor\u00eda Primera B (2)" (\n"season" TEXT,\n  "champion ( title count )" TEXT,\n  "runner-up" TEXT,\n  "third place" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./Categor\u00eda Primera B (2)" LIMIT 3\nseason champion ( title count )               runner-up      third place\n  1991           envigado ( 1 )          alianza llanos   atl\u00e9tico huila\n  1992     atl\u00e9tico huila ( 1 )          alianza llanos         cortulu\u00e1\n  1993           cortulu\u00e1 ( 1 ) fiorentina de florencia atl\u00e9tico palmira\n*/\n\nCREATE TABLE "./Primera B Nacional (1)" (\n"team" TEXT,\n  "titles" INTEGER,\n  "years won" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./Primera B Nacional (1)" LIMIT 3\n    team  titles                  years won\nbanfield       3 1992-93 , 2000-1 , 2013-14\n  olimpo       3  2001-2 , 2006-7 , 2009-10\n hurac\u00e1n       2           1989-90 , 1999-0\n*/',
        "bridge_hints": "",
        "question": "Which Primera B Nacional team finished second in the year the club founded on 21 January 1896 finished third ?",
        "blendsql": """
        SELECT "runner-up" FROM "./Primera B Nacional (0)" AS w
        WHERE "third place" = {{
            LLMQA(
                'Which club was founded on 21 January 1896?'
                (SELECT * FROM documents WHERE documents MATCH 'primera OR founded OR (club AND 1896)' ORDER BY rank LIMIT 5)
                options='w::third place'
            )
        }}
        """,
    },
    {
        "serialized_db": 'CREATE TABLE "./List of African films (4)" (\n"year" INTEGER,\n  "title" TEXT,\n  "director" TEXT,\n  "genre" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./List of African films (4)" LIMIT 3\n year                        title         director             genre\n 1972                       kouami metonou do kokou             short\n 1979 au rendez-vous du r\u00eave ab\u00eati  kodjo goncalves short documentary\n 1986        the blooms of banjeli   carlyn saltman short documentary\n*/\n\nCREATE TABLE "./Cinema of Chad (0)" (\n"year" INTEGER,\n  "title" TEXT,\n  "director" TEXT,\n  "genre" TEXT,\n  "notes" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./Cinema of Chad (0)" LIMIT 3\n year                                         title                   director              genre                                notes\n 1958                           the roots of heaven                john huston     drama , action    american film partly shot in chad\n 1960 les tonnes de l\'audace - mission t\u00e9n\u00e9r\u00e9 tchad ren\u00e9 quinet & louis sommet        documentary                                 none\n 1966                             p\u00eacheurs du chari             edouard sailly ethnographic short english title : fishers of the chari\n*/\n\nCREATE TABLE "./Cinema of Chad (1)" (\n"year" INTEGER,\n  "title" TEXT,\n  "director" TEXT,\n  "genre" TEXT,\n  "notes" TEXT\n)\n/*\n3 example rows:\nSELECT * FROM "./Cinema of Chad (1)" LIMIT 3\n year                                         title                   director              genre                                notes\n 1958                           the roots of heaven                john huston     drama , action    american film partly shot in chad\n 1960 les tonnes de l\'audace - mission t\u00e9n\u00e9r\u00e9 tchad ren\u00e9 quinet & louis sommet        documentary                                 none\n 1966                             p\u00eacheurs du chari             edouard sailly ethnographic short english title : fishers of the chari\n*/',
        "bridge_hints": "",
        "question": "Who is the director the Togolese film that was a 30 minute film that was shot in 16mm ?",
        "blendsql": """
        SELECT director FROM "./List of African films (4)" as w
        WHERE title = {{
            LLMQA(
                'What is the name of the Togolese film that was 30 minutes and shot in 16mm?'
                (SELECT * FROM documents WHERE documents MATCH 'togolese OR 30 OR 16mm OR film' ORDER BY rank LIMIT 5)
                options='w::title'
            )
        }}
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
