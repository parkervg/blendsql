blendsql_examples = [
    {
    "serialized_db": """
    Table Description: Andy Karl | Awards and nominations
    CREATE TABLE w (index INTEGER,
                year INTEGER,
                award TEXT,
                category TEXT,
                work TEXT,
                result TEXT)
    /* 
    3 example rows:
    SELECT * FROM w LIMIT 3 
    index  year                        award                                    category                       work    result
     0  2013             drama desk award     outstanding featured actor in a musical the mystery of edwin drood nominated
     1  2013 broadway.com audience awards favorite onstage pair (with jessie mueller) the mystery of edwin drood nominated
     2  2014                   tony award                     best actor in a musical                      rocky nominated
    */
    """,
        "bridge_hints": "",
        "question": "When did Andy Karl win the Olivier Award and for which of his work?",
        "blendsql": """
    {{
        LLMQA(
            'Describe how did Andy Karl win his Olivier Award?',
                (
                SELECT * FROM w WHERE award like '%Olivier Award');
                )
            )
    }}
    """,
    },
    {
    "serialized_db": """
    Table Description: Pooja Ramachandran | Filmography
    CREATE TABLE w (
    index INTEGER,
      year INTEGER,
      film TEXT,
      role TEXT,
      language TEXT,
      notes REAL
    )
    /*
    3 example rows:
    SELECT * FROM w LIMIT 3
     index  year                           film  role  language notes
         0  2002         yathrakarude sradhakku  none malayalam  None
         1  2012 kadhalil sodhappuvadhu yeppadi cathy     tamil  None
         2  2012                   love failure cathy    telugu  None
    */
    """,
        "bridge_hints": "",
        "question": "In what films did Pooja Ramachandran play Cathy?",
        "blendsql": """
        {{
                LLMQA(
                    'Describe the film in which Pooja Ramachandran played the role of Cathy?',
                    (
                     SELECT * FROM w WHERE role='cathy';
                     )
                )
        }}    
    """,
    },
    {
    "serialized_db": """
    Table Description: Dyro | Awards and nominations
    CREATE TABLE w (
    index INTEGER,
      year INTEGER,
      award TEXT,
      nominee TEXT,
      category TEXT,
      result INTEGER
    )
    /*
    3 example rows:
    SELECT * FROM w LIMIT 3
     index  year              award nominee    category  result
         0  2013 dj magazine awards    dyro top 100 djs      30
         1  2014 dj magazine awards    dyro top 100 djs      27
         2  2015 dj magazine awards    dyro top 100 djs      27
    */
    """,
        "bridge_hints": "",
        "question": "Dyro ranked how high and in what category for what award in 2014?",
        "blendsql": """
    {{  
        LLMQA(
            'Describe the performance and award of Dyro ranked in the year of 2014', 
                (
                    SELECT * FROM w WHERE nominee = 'dyro' AND year = '2014';
                )
            )
    }}
    """,
    },
{
    "serialized_db": """
    Table Description: List of best-selling albums in Japan | List of best-selling albums by domestic acts
    CREATE TABLE w (
    index INTEGER,
    no. INTEGER,
    album TEXT,
    artist TEXT,
    released TEXT,
    chart INTEGER,
    sales INTEGER
)

/*
3 example rows:
SELECT * FROM w LIMIT 3
 index  no.                   album       artist  released  chart   sales
     0    1              first love hikaru utada 1999-3-10      1 7672000
     1    2 b'z the best "pleasure"          b'z 1998-5-20      1 5136000
     2    3                  review         glay 1997-10-1      1 4876000
*/
    """,
        "bridge_hints": "",
        "question": "How many copies did Pleasure sell in 1998 alone, and how long was it the best selling album in Japan?",
        "blendsql": """
    {{  
        LLMQA(
            'In 1998, how many copies was Pleasure sold, and how long was it the best selling album in Japan?', 
                (
                    SELECT * FROM w where released >= (select released from w where album = 'b'z the best 'pleasure'');
                )
            )
    }}
    """,
    },
{
    "serialized_db": """
    Table Description: Ben Platt (actor) | Theatre credits
    CREATE TABLE w (
    index INTEGER,
    year TEXT,
    production TEXT,
    role TEXT,
    venue TEXT,
    notes TEXT
)

/*
3 example rows:
SELECT * FROM w LIMIT 3
 index year          production            role            venue         notes
     0 2002       the music man  winthrop paroo   hollywood bowl   los angeles
     1 2004 caroline, or change    noah gellman ahmanson theatre national tour
     2 2005            dead end philip griswald ahmanson theatre      regional
*/
    """,
        "bridge_hints": "",
        "question": "When and in what play did Platt appear at the Music Box Theatre?",
        "blendsql": """
    {{  
        LLMQA(
            "Describe Platt's performance at the Music Box Theatre",  
                (
                    SELECT * FROM w where venue = 'Music Box Theatre';
                )
            )
    }}
    """,
    },
{
    "serialized_db": """
    Table Description: E-UTRA | User Equipment (UE) categories
    CREATE TABLE w (
    "index" INTEGER,
    "user equipment category" TEXT,
    "max. l1 data rate downlink (mbit/s)" REAL,
    "max. number of dl mimo layers" TEXT,
    "max. l1 data rate uplink (mbit/s)" TEXT,
    "3gpp release" TEXT
)

/*
3 example rows:
SELECT * FROM w LIMIT 3
 index user equipment category  max. l1 data rate downlink (mbit/s) max. number of dl mimo layers max. l1 data rate uplink (mbit/s) 3gpp release
     0                     nb1                                 0.68                             1                                 1       rel 13
     1                      m1                                 1.00                             1                                 1       rel 13
     2                       0                                 1.00                             1                                 1       rel 12
*/
    """,
        "bridge_hints": "",
        "question": "What are the download rates of EUTRAN?",
        "blendsql": """
    {{  
        LLMQA(
            "Describe the internet speed of EUTRAN",  
                (
                    SELECT "max. l1 data rate downlink (mbit/s)" and "max. l1 data rate uplink (mbit/s)" FROM w;
                )
            )
    }}
    """,
    },
{
    "serialized_db": """
    Table Description: Austin Fyten | Career statistics
    CREATE TABLE "w" (
    "index" INTEGER,
    "-" TEXT,
    "-_2" TEXT,
    "-_3" TEXT,
    "regular season" REAL,
    "regular season_2" TEXT,
    "regular season_3" TEXT,
    "regular season_4" TEXT,
    "regular season_5" TEXT,
    "playoffs" TEXT,
    "playoffs_2" REAL,
    "playoffs_3" TEXT,
    "playoffs_4" TEXT,
    "playoffs_5" TEXT,
    "-_4" TEXT,
    "-_5" TEXT
)

/*
3 example rows:
SELECT * FROM "w" LIMIT 3
 index      -            -_2    -_3 regular season regular season_2 regular season_3 regular season_4 regular season_5 playoffs playoffs_2 playoffs_3 playoffs_4 playoffs_5 -_4 -_5
     0 season           team league           None               gp                g                a              pts      pim       None         gp          g          a pts pim
     1 2004–5 airdrie xtreme  ambhl           None               37               19               20               39       28       None          4          2          3   5   6
     2 2005–6 airdrie xtreme  ambhl           None               34               23               23               46       62       None          4          3          5   8   8
*/
    """,
        "bridge_hints": "",
        "question": "What two teams did Austin Fyten play for during the 2015-16 season, and what league was the first team in?",
        "blendsql": """
    {{  
        LLMQA(
            'Describe the team and league information of Austin Fyten during the 2015-16 season.', 
                (
                    SELECT team, league FROM w where season = '2015-16';
                )
            )
    }}
    """,
    },
{
"serialized_db": """
Table Description: World U-17 Hockey Challenge | Results
CREATE TABLE "w" (
"index" INTEGER,
  "year" TEXT,
  "gold" TEXT,
  "silver" TEXT,
  "bronze" TEXT,
  "host city (cities)" TEXT
)
/*
3 example rows:
SELECT * FROM "w" LIMIT 3
 index year          gold     silver         bronze                                  host city (cities)
     0 2019          none       none           none alberta medicine hat and saskatchewan swift current
     1 2018        russia    finland         sweden             new brunswick quispamsis and saint john
     2 2017 united states canada red czech republic     british columbia dawson creek and fort st. john
*/
""",
    "bridge_hints": "",
    "question": "What countries did the World U-17 Hockey Challenge attract after 2016?",
    "blendsql": """
{{  
    LLMQA(
        'Summarize the country participating the World U-17 Hockey Challenge after 2016', 
            (
                SELECT *  FROM w WHERE year>2016';
            )
        )
}}
""",
},
{
        "serialized_db": """
    Table Description: Tigerair Australia | Fleet
    CREATE TABLE w (
    index INTEGER,
    aircraft TEXT,
    in service INTEGER,
    orders TEXT,
    passengers REAL,
    notes TEXT
)
/*
3 example rows:
SELECT * FROM w LIMIT 3
 index        aircraft  in service orders  passengers                                                                                                   notes
     0 airbus a320-200          10      —       180.0 all to be replaced by boeing 737-800. aircraft to be transferred to virgin australia regional airlines.
     1  boeing 737-800           5      —       186.0                                                              aircraft transferred from virgin australia
     2           total          15   none         NaN                                                                                                    none
*/
    """,
        "bridge_hints": "",
        "question": "How many passengers can that plane hold?",
        "blendsql": """
    {{  
        LLMQA(
            'How many passengers can that plane hold?', 
                (
                    SELECT * FROM w;
                )
            )
    }}
    """,
    },
{
        "serialized_db": """
    Table Description: 1982 Illinois gubernatorial election | Results
    CREATE TABLE w (
    index INTEGER,
    party TEXT,
    party_2 TEXT,
    candidate TEXT,
    votes TEXT,
    % REAL,
    ± TEXT
)
/*
3 example rows:
SELECT * FROM w LIMIT 3
 index party     party_2                     candidate   votes     %    ±
     0  none  republican james r. thompson (incumbent) 1816101 49.44 none
     1  none  democratic           adlai stevenson iii 1811027 49.30 none
     2  none libertarian                 bea armstrong   24417  0.66 none
*/
    """,
        "bridge_hints": "",
        "question": "Who won the 1982 Illinois gubernatorial election, and how many votes was the margin?",
        "blendsql": """
    {{  
        LLMQA(
            'Describe the performance of winner in 1982 Illinois gubernatorial election', 
                (
                    SELECT * FROM w;
                )
            )
    }}
    """,
    },
]
sql_examples = []
