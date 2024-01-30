# BlendSQL Research

Example usage:
```
caffeinate -is python -m research.run-evaluate research/configs/gpt-4-all.json
```
## Initial Results

Below experiments were done on the first 200 items in the WikiTableQuestions validation set.

They included only 4 few-shot examples found in [./prompts/few_shot.py](./prompts/few_shot.py).

Full model outputs are in [./outputs](./outputs).

| **Parser**       | **QA**           | **Accuracy** | **# With Ingredients** |
|------------------|------------------|--------------|------------------------|
| gpt-4            | gpt-4            | 67.35        | 6                      |
| gpt-4            | text-davinci-003 | 66.84        | 6                      |
| text-davinci-003 | text-davinci-003 | 45.65        | 53                     |
| text-davinci-003 | gpt-4            | 49.46        | 53                     |

Some can be solved with both BlendSQL and SQLite
"what is the largest penalty?"

`SELECT MAX(CAST(SUBSTR("Penalties (P+P+S+S)", 0, 3) AS integer)) FROM w;`
`SELECT MAX({{LLMMap('total penalties?', 'w::penalties (p+p+s+s)')}}) FROM w`


WizardCoder + BlendSQL on whole WikiTQ dev set:
```json
{
    "metric_score": 0.46872166817769717,
    "num_with_ingredients": 27,
    "num_errors": 42,
    "num_completed": 2164,
    "time_elapsed": 489.1062750816345
}
```

## Helpful Tips

- Use `max_val_samples` for debugging, to restrict how many datapoints we iterate over during a `do_eval` loop.
- To add a dataset, do the following:
  1) Add a dataset script to [datasets/](./datasets)
     1) See [./datasets/wikitq.py](./datasets/wikitq/wikitq.py) for an example
  2) Add a metric script to [metrics/](./metrics)
     1) See [./metrics/wikitq.py](./metrics/wikitq/wikitq.py) for an example
  3) Put all dataset-specif utils in `./utils/{dataset_name}`
     1) As done in [./utils/wikitq/wikitq.py](./utils/wikitq/wikitq.py), this should also contain a pre-process function    
  4) Modify `load_dataset()` in [./utils/dataset_loader.py](./utils/dataset_loader.py) with dataset-specific logic


## General Notes
- GPT-4 is much more capable at discerning when we need BlendSQL, vs. can just use native SQL
  - Examples:
    - using `SUBSTR` calls
    - Leveraging table title, like if we have 'Hospitals in California' table, we don't need a condition `state = 'CA'`
- Using no `binary` argument in the LLM MapIngredient call improves results
  - On initial 200:
    - 0.479 vs 0.463 with text-davinci-003 as parser, gpt-4 as QA
    - 0.432 vs 0.406 with text-davinci-003 as parser and QA
  - This avoids the catastrophic parsing issue where we fail to include `binary=0` in our ingredient call
  - But, puts more pressure on the QA model to be right

The new `LLMQA` ingredient works, but we have some issues with response formatting.

https://www.sqlite.org/fts3.html#appendix_a
"In SQLite, when a sub-query used in a join contains a LIMIT clause, the results of the sub-query are calculated and stored in temporary table before the main query is executed."

https://neuml.hashnode.dev/building-an-efficient-sparse-keyword-index-in-python

```python 
metric.add(
    prediction=["The same problem was oil pressure."],
    reference={
        "answer_text": ["Oil pressure"],
        "question": "what was the same problem that bernard collomb had as innes ireland?",
    },
)
```
The above WikiTQ metric computation results in a score of 0.0
We could either do some post-processing (e.g. Bridge to extract 'oil pressure') or better prompting.

Could benefit from some external tools, such as below
For 'difference of points'

```json
 "sql": "SELECT MAX ( abs ( `Result` - `Result` ) ) FROM w",
"answer_text": [
    "29"
],
"input_program_args": {
    "bridge_hints": "",
    "question": "what was the largest difference of points in a single game?",
    "serialized_db": "Table Description: 1960 Buffalo Bills season\nCREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"week\" INTEGER,\n  \"date\" TEXT,\n  \"opponent\" TEXT,\n  \"result\" TEXT,\n  \"game site\" TEXT,\n  \"attendance\" INTEGER\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id  week      date           opponent  result            game site  attendance\n     0       0     1 1960-9-11 at new york titans  l 27\u20133         polo grounds       10200\n     1       1     2 1960-9-18     denver broncos l 27\u201321 war memorial stadium       15229\n     2       2     3 1960-9-23 at boston patriots  w 13\u20130      nickerson field       20732\n*/"
},
"db_path": "research/db/wikitq/csv/203-csv/775.db",
"pred_text": [
    "8"
],
"pred_sql": "SELECT MAX({{LLMMap('difference of points', 'w::result')}}) FROM w",
"pred_has_ingredient": true,
```

## Good Examples
- [Example 1](#good_example_1)
- [Example 2](#good_example_2)
- [Example 3](#good_example_3)
- [Example 4](#good_example_4)
- [Example 5](#good_example_5)

### <a id="good_example_1"></a> Good Example 1
```json
{
    "id": "nt-267",
    "question": "total number of players whose home town was in north carolina (nc)",
    "table_id": "csv/204-csv/526.csv",
    "table": {
        "page_title": "2010\u201311 North Carolina Tar Heels men's basketball team",
        "header": [
            "Name",
            "#",
            "Position",
            "Height",
            "Weight",
            "Year",
            "Home Town",
            "High School"
        ],
        "rows": [
            [
                "Harrison Barnes",
                "40",
                "Forward",
                "6\u20138",
                "210",
                "Freshman",
                "Ames, IA",
                "Ames"
            ],
            [
                "Daniel Bolick",
                "3",
                "Guard",
                "5\u201310",
                "175",
                "Senior",
                "Carrboro, NC",
                "Chapel Hill"
            ],
            [
                "Reggie Bullock",
                "35",
                "Guard",
                "6\u20137",
                "190",
                "Freshman",
                "Kinston, NC",
                "Kinston"
            ],
            [
                "Stewart Cooper",
                "15",
                "Forward",
                "6\u20135",
                "205",
                "Junior",
                "Winston-Salem, NC",
                "Forsyth County Day"
            ],
            [
                "Patrick Crouch",
                "30",
                "Guard",
                "5\u201311",
                "175",
                "Junior",
                "Asheville, NC",
                "T.C. Roberson"
            ],
            [
                "Larry Drew II*",
                "11",
                "Guard",
                "6\u20132",
                "180",
                "Junior",
                "Encino, CA",
                "Woodland Hills Taft"
            ],
            [
                "David Dupont",
                "22",
                "Forward",
                "6\u20135",
                "195",
                "Junior",
                "Greensboro, NC",
                "Grimsley"
            ],
            [
                "Van Hatchell",
                "13",
                "Forward",
                "6\u20134",
                "185",
                "Senior",
                "Chapel Hill, NC",
                "Cresset Christian"
            ],
            [
                "John Henson",
                "31",
                "Forward",
                "6\u201310",
                "210",
                "Sophomore",
                "Tampa, FL",
                "Sickles"
            ],
            [
                "D.J. Johnston",
                "32",
                "Forward",
                "6\u20134",
                "195",
                "Junior",
                "Lower Gwynedd, PA",
                "Germantown"
            ],
            [
                "Justin Knox",
                "25",
                "Forward",
                "6\u20139",
                "240",
                "Graduate",
                "Tuscaloosa, AL",
                "Central"
            ],
            [
                "Kendall Marshall",
                "5",
                "Guard",
                "6\u20133",
                "186",
                "Freshman",
                "Dumfries, VA",
                "Bishop O'Connell"
            ],
            [
                "Leslie McDonald",
                "15",
                "Guard",
                "6\u20134",
                "215",
                "Sophomore",
                "Memphis, TN",
                "Briarcrest Christian"
            ],
            [
                "Dexter Strickland",
                "1",
                "Guard",
                "6\u20133",
                "180",
                "Sophomore",
                "Rahway, NJ",
                "St. Patrick"
            ],
            [
                "Justin Watts",
                "24",
                "Guard",
                "6\u20134",
                "210",
                "Junior",
                "Durham, NC",
                "Jordan"
            ],
            [
                "Tyler Zeller",
                "44",
                "Forward",
                "7\u20130",
                "250",
                "Junior",
                "Washington, IN",
                "Washington"
            ]
        ]
    },
    "sql": "SELECT COUNT ( `Name` ) FROM w WHERE `Home Town` = 'nc'",
    "answer_text": [
        "7"
    ],
    "input_program_args": {
        "bridge_hints": "",
        "examples": [
            {
                "blendsql": "SELECT DISTINCT description, merchant FROM transactions WHERE\n    {{LLM('is this a pizza shop?', 'transactions::merchant', endpoint_name='gpt-4')}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'transactions::description',\n           binary=0,\n           example_outputs='TX;CA;MA;-1;'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                "question": "Pizza shops in MA",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            },
            {
                "blendsql": "SELECT DISTINCT description, merchant, child_category, date FROM transactions WHERE\n      {{LLM('would this gift be appropriate for a child?', 'transactions::description')}} = 1\n      AND child_category = 'Gifts'",
                "question": "Show gifts for my kid",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            }
        ],
        "program": "\n{{#system~}}\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{~/system}}\n\n{{#user~}}\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL:\n{{~/user}}\n\n{{#assistant~}}\n{{gen \"result\" temperature=0.0}}\n{{~/assistant}}\n",
        "question": "total number of players whose home town was in north carolina (nc)",
        "serialized_db": "CREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"name\" TEXT,\n  \"#\" INTEGER,\n  \"position\" TEXT,\n  \"height\" TEXT,\n  \"weight\" INTEGER,\n  \"year\" TEXT,\n  \"home town\" TEXT,\n  \"high school\" TEXT\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id            name  # position height  weight     year    home town high school\n     0       0 harrison barnes 40  forward    6\u20138     210 freshman     ames, ia        ames\n     1       1   daniel bolick  3    guard   5\u201310     175   senior carrboro, nc chapel hill\n     2       2  reggie bullock 35    guard    6\u20137     190 freshman  kinston, nc     kinston\n*/"
    },
    "db_path": "./research/tmp/8fa07b3b-f51d-4690-ad04-e908fe0cf071.db",
    "pred_text": [
        7
    ],
    "pred_sql": "SELECT COUNT(*) FROM w WHERE {{LLM('is this town in North Carolina?', 'w::home town')}} = 1"
}
```
### <a id="good_example_2"></a> Good Example 2
```json 
{
    "id": "nt-647",
    "question": "in how many games did sri lanka score at least 2 goals?",
    "table_id": "csv/204-csv/913.csv",
    "table": {
        "page_title": "Kasun Jayasuriya",
        "header": [
            "#",
            "Date",
            "Venue",
            "Opponent",
            "Score",
            "Result",
            "Competition"
        ],
        "rows": [
            [
                "1.",
                "1 May 2000",
                "Rasmee Dhandu Stadium, Mal\u00e9, Maldives",
                "Maldives",
                "1\u20131",
                "1\u20131",
                "2000 MFF Golden Jubilee Tournament"
            ],
            [
                "2.",
                "25 August 2000",
                "Ho Chi Minh City, Vietnam",
                "Vietnam",
                "2\u20131",
                "2\u20132",
                "2000 HCM Cup"
            ],
            [
                "3.",
                "28 March 2001",
                "Colombo, Sri Lanka",
                "Cambodia",
                "1\u20130",
                "1\u20130",
                "Friendly"
            ],
            [
                "4.",
                "17 May 2001",
                "Beirut Municipal Stadium, Beirut, Lebanon",
                "Pakistan",
                "3\u20132",
                "3\u20133",
                "2002 FIFA World Cup qualification (AFC)"
            ],
            [
                "5.",
                "30 May 2001",
                "Suphachalasai Stadium, Bangkok, Thailand",
                "Pakistan",
                "1\u20130",
                "3\u20131",
                "2002 FIFA World Cup qualification (AFC)"
            ],
            [
                "6.",
                "30 May 2001",
                "Supachalasai Stadium, Bangkok, Thailand",
                "Pakistan",
                "3\u20130",
                "3\u20131",
                "2002 FIFA World Cup qualification (AFC)"
            ],
            [
                "7.",
                "4 April 2002",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Maldives",
                "1\u20130",
                "1\u20130",
                "Friendly"
            ],
            [
                "8.",
                "21 March 2003",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Timor-Leste",
                "1\u20131",
                "3\u20132",
                "2004 AFC Asian Cup qualification"
            ],
            [
                "9.",
                "21 March 2003",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Timor-Leste",
                "3\u20132",
                "3\u20132",
                "2004 AFC Asian Cup qualification"
            ],
            [
                "10.",
                "3 December 2003",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Laos",
                "2\u20130",
                "3\u20130",
                "2006 FIFA World Cup qualification (AFC)"
            ],
            [
                "11.",
                "2 April 2006",
                "MA Aziz Stadium, Chittagong, Bangladesh",
                "Brunei",
                "2\u20130",
                "3\u20130",
                "2006 AFC Challenge Cup"
            ],
            [
                "12.",
                "12 April 2006",
                "MA Aziz Stadium, Chittagong, Bangladesh",
                "Nepal",
                "1\u20130",
                "1\u20131",
                "2006 AFC Challenge Cup"
            ],
            [
                "13.",
                "24 March 2007",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Malaysia",
                "1\u20134",
                "1\u20134",
                "Friendly"
            ],
            [
                "14.",
                "2 April 2008",
                "Chungshan Stadium, Taipei, Taiwan",
                "Guam",
                "5\u20131",
                "5\u20131",
                "2008 AFC Challenge Cup qualification"
            ],
            [
                "15.",
                "4 April 2008",
                "Chungshan Stadium, Taipei, Taiwan",
                "Pakistan",
                "1\u20130",
                "7\u20131",
                "2008 AFC Challenge Cup qualification"
            ],
            [
                "16.",
                "4 April 2008",
                "Chungshan Stadium, Taipei, Taiwan",
                "Pakistan",
                "4\u20131",
                "7\u20131",
                "2008 AFC Challenge Cup qualification"
            ],
            [
                "17.",
                "4 April 2008",
                "Chungshan Stadium, Taipei, Taiwan",
                "Pakistan",
                "7\u20131",
                "7\u20131",
                "2008 AFC Challenge Cup qualification"
            ],
            [
                "18.",
                "6 April 2008",
                "Chungshan Stadium, Taipei, Taiwan",
                "Chinese Taipei",
                "2\u20132",
                "2\u20132",
                "2008 AFC Challenge Cup qualification"
            ],
            [
                "19.",
                "2 August 2008",
                "Gachibowli Athletic Stadium, Hyderabad, India",
                "Myanmar",
                "1\u20131",
                "1\u20133",
                "2008 AFC Challenge Cup"
            ],
            [
                "20.",
                "4 April 2009",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Brunei",
                "1\u20130",
                "5\u20131",
                "2010 AFC Challenge Cup qualification"
            ],
            [
                "21.",
                "4 April 2009",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Brunei",
                "3\u20130",
                "5\u20131",
                "2010 AFC Challenge Cup qualification"
            ],
            [
                "22.",
                "4 April 2009",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Brunei",
                "4\u20130",
                "5\u20131",
                "2010 AFC Challenge Cup qualification"
            ],
            [
                "23.",
                "4 April 2009",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Brunei",
                "5\u20130",
                "5\u20131",
                "2010 AFC Challenge Cup qualification"
            ],
            [
                "24.",
                "6 April 2009",
                "Sugathadasa Stadium, Colombo, Sri Lanka",
                "Chinese Taipei",
                "1\u20130",
                "2\u20131",
                "2010 AFC Challenge Cup qualification"
            ],
            [
                "25.",
                "6 December 2009",
                "Bangabandhu National Stadium, Dhaka, Bangladesh",
                "Bhutan",
                "3\u20130",
                "6\u20130",
                "2009 South Asian Football Federation Cup"
            ],
            [
                "26.",
                "6 December 2009",
                "Bangabandhu National Stadium, Dhaka, Bangladesh",
                "Bhutan",
                "4\u20130",
                "6\u20130",
                "2009 South Asian Football Federation Cup"
            ],
            [
                "27.",
                "6 December 2009",
                "Bangabandhu National Stadium, Dhaka, Bangladesh",
                "Bhutan",
                "5\u20130",
                "6\u20130",
                "2009 South Asian Football Federation Cup"
            ]
        ]
    },
    "sql": "SELECT COUNT ( * ) FROM w WHERE `Score` >= 2",
    "answer_text": [
        "16"
    ],
    "input_program_args": {
        "bridge_hints": "",
        "examples": [
            {
                "blendsql": "SELECT DISTINCT description, merchant FROM transactions WHERE\n    {{LLM('is this a pizza shop?', 'transactions::merchant', endpoint_name='gpt-4')}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'transactions::description',\n           binary=0,\n           example_outputs='TX;CA;MA;-1;'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                "question": "Pizza shops in MA",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            },
            {
                "blendsql": "SELECT DISTINCT description, merchant, child_category, date FROM transactions WHERE\n      {{LLM('would this gift be appropriate for a child?', 'transactions::description')}} = 1\n      AND child_category = 'Gifts'",
                "question": "Show gifts for my kid",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            }
        ],
        "program": "\n{{#system~}}\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{~/system}}\n\n{{#user~}}\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL:\n{{~/user}}\n\n{{#assistant~}}\n{{gen \"result\" temperature=0.0}}\n{{~/assistant}}\n",
        "question": "in how many games did sri lanka score at least 2 goals?",
        "serialized_db": "CREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"#\" REAL,\n  \"date\" TEXT,\n  \"venue\" TEXT,\n  \"opponent\" TEXT,\n  \"score\" TEXT,\n  \"result\" TEXT,\n  \"competition\" TEXT\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id   #      date                                 venue opponent score result                        competition\n     0       0 1.0  2000-5-1 rasmee dhandu stadium, mal\u00e9, maldives maldives   1\u20131    1\u20131 2000 mff golden jubilee tournament\n     1       1 2.0 2000-8-25             ho chi minh city, vietnam  vietnam   2\u20131    2\u20132                       2000 hcm cup\n     2       2 3.0 2001-3-28                    colombo, sri lanka cambodia   1\u20130    1\u20130                           friendly\n*/"
    },
    "db_path": "./research/tmp/4bd65c0e-5640-42e8-8af9-5274ea1f36d4.db",
    "pred_text": [
        16
    ],
    "pred_sql": "SELECT COUNT(*) FROM w WHERE {{LLM('how many goals did sri lanka score in this game?', 'w::score')}} >= 2"
}
```

### <a id="good_example_3"></a> Good Example 3
```json
{
    "id": "nt-373",
    "question": "tell me the number of shows that only have one host per day.",
    "table_id": "csv/204-csv/839.csv",
    "table": {
        "page_title": "List of programs broadcast by Fox News Channel",
        "header": [
            "ET",
            "Days available",
            "Program",
            "Host(s)",
            "Description"
        ],
        "rows": [
            [
                "9am\u201310am",
                "Monday\u2013Friday",
                "The Morning Click",
                "Jamie Colby (M,T), Harris Faulkner (W\u2013F)",
                "Current events."
            ],
            [
                "10am\u201310:30am",
                "Monday\u2013Friday",
                "",
                "Lauren Green (M), Uma Pemmaraju (T), Gregg Jarrett (W), Arthel Neville (F), Rick Folbaum (F), Heather Childers",
                "Current events."
            ],
            [
                "11am\u201311:30am",
                "Monday\u2013Friday",
                "FBN Live",
                "Lauren Simonetti",
                "Business news and information."
            ],
            [
                "12pm\u20131pm",
                "Monday\u2013Friday",
                "On the Hunt",
                "Jonathan Hunt",
                "Current events."
            ],
            [
                "2pm\u20133pm",
                "Monday",
                "Entertainment Hour",
                "Ashley Dvorkin",
                "Entertainment news, musical performances."
            ],
            [
                "2pm\u20133pm",
                "Tuesday",
                "",
                "Kimberly Guilfoyle",
                "Crime and legal news."
            ],
            [
                "2pm\u20133pm",
                "Wednesday",
                "Defcon 3",
                "Kathleen Troia McFarland",
                "Military news and information."
            ],
            [
                "2pm\u20133pm",
                "Thursday",
                "Fox Car Report",
                "Gary Gastelu",
                "Automotive news and information."
            ],
            [
                "2pm\u20133pm",
                "Friday",
                "Spirited Debate",
                "Lauren Green",
                "Discussion on religious topics."
            ]
        ]
    },
    "sql": "SELECT COUNT ( `Program` ) FROM w WHERE `Host(s)` = 1",
    "answer_text": [
        "7"
    ],
    "input_program_args": {
        "bridge_hints": "",
        "examples": [
            {
                "blendsql": "SELECT year FROM w WHERE opponent = 'elizabeth matory'",
                "question": "What year was Elizabeth Matory the opponent of Charles Albert Ruppersberger?",
                "serialized_db": "\nTable Description: Dutch Ruppersberger (Electoral history)\nCREATE TABLE w(\n\trow_id int,\n\tyear int,\n\toffice text,\n\telection text,\n\tfilledcolumnname real,\n\tsubject text,\n\tparty text,\n\tvotes text,\n\t% text,\n\tfilledcolumnname_2 real,\n\topponent text,\n\tparty_2 text,\n\tvotes_2 text,\n\t%_2 text)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3;\nrow_id\tyear\toffice\telection\tfilledcolumnname\tsubject\tparty\tvotes\t%\tfilledcolumnname_2\topponent\tparty_2\tvotes_2\t%_2\n0\t1994\tbaltimore county executive\tgeneral\tnan\tdutch ruppersberger\tdemocratic\tn/a\tn/a\tnan\tn/a\tn/a\tn/a\tn/a\n1\t1998\tbaltimore county executive\tgeneral\tnan\tdutch ruppersberger\tdemocratic\t166482\t70.47\tnan\tjohn j. bishop\trepublican\t69449\t29.4\n2\t2002\tnone\tgeneral\tnan\tdutch ruppersberger\tdemocratic\t105718\t54.16\tnan\thelen delich bentley\trepublican\t88954\t45.57\n*/\n"
            },
            {
                "blendsql": "SELECT system FROM w WHERE system IN ('game boy advance', 'msx (japan only)') ORDER BY japan LIMIT 1",
                "question": "Which system has a lower number for Japan of the virtual console systems: Game Boy Advance or the Japan-only console MSX?",
                "serialized_db": "\nTable Description: Virtual Console (Titles)\nCREATE TABLE w (\n\trow_id int,\n\tsystem text,\n\tjapan int,\n\t[[list of virtual console games for wii u (north america)|north  america]] real,\n\tpal region - europe real,\n\tpal region - australia real\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3;\nrow_id\tsystem\tjapan\t[[list of virtual console games for wii u (north america)|north  america]]\tpal region - europe\tpal region - australia\n0\tnes/famicom\t148\t94.0\t89.0\t89.0\n1\tsuper nes/super famicom\t101\t51.0\t49.0\t49.0\n2\tnintendo 64\t22\t21.0\t21.0\t21.0\n*/  \n            "
            },
            {
                "blendsql": "SELECT player FROM w WHERE \n            {{\n                LLM('Has crossed swords on its logo?'; 'w::gold coast titans')\n            }} = 1\n        ",
                "question": "What player was transferred from the team that has crossed swords on its logo to the Warrington Wolves in the 2018 season?",
                "serialized_db": "\nTable Description: 2018 Warrington Wolves season (Transfers | In)\nCREATE TABLE w(\n\trow_id int,\n\tplayer text,\n\tsigned from text,\n\tcontract length text,\n\tannounced text)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3;\nrow_id\tplayer\tsigned from\tcontract length\tannounced\n0\tsitaleki akauola\tpenrith panthers\tp2y\t2017-08-01 00:00:00\n1\tbryson goodwin\tsouth sydney rabbitohs\tp2y\t2017-10-01 00:00:00\n2\ttyrone roberts\tgold coast titans\tp3y\t2017-10-01 00:00:00\n*/\nCREATE TABLE Images(\n\trow_id int,\n\tgold coast titans text)\n/*\nAll rows of the table:\nSELECT * FROM w;\nrow_id\tgold coast titans\n0\ta logo for the golden knights is painted on the beach.\n*/\n"
            },
            {
                "blendsql": "SELECT DISTINCT merchant FROM w WHERE\n    {{LLM('is this a pizza shop?', 'w::merchant'}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'w::description',\n           example_outputs='TX;CA;MA;-1;'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                "question": "Pizza shop merchants in MA",
                "serialized_db": "\nTable Description: transactions\nCREATE TABLE w (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n/*\n3 example rows:\nSELECT * FROM transactions LIMIT 3\n index                                    description  amount merchant cash_flow parent_category         child_category       date  confidence_score\n     0  Zelle payment from JUSTIN ORENIC BACucw5n1t14  2000.0    Zelle   Expense        Cash/ATM Online Payment Service 2022-06-20          0.698891\n     1 Zelle Transfer Conf  UJ7QJF7TE  Emilia Galindo  1500.0    Zelle   Expense        Cash/ATM Online Payment Service 2022-01-16          0.701658\n     2   Zelle payment from SOPHIA YANEZ WFCT0Q9R3CNR  1474.0    Zelle   Expense        Cash/ATM Online Payment Service 2022-03-27          0.861237\n*/\n)"
            }
        ],
        "program": "\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL: {{gen \"result\" temperature=0.0}}\n",
        "question": "tell me the number of shows that only have one host per day.",
        "serialized_db": "Table Description: List of programs broadcast by Fox News Channel\nCREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"et\" TEXT,\n  \"days available\" TEXT,\n  \"program\" TEXT,\n  \"host(s)\" TEXT,\n  \"description\" TEXT\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id                 et              days available       program                                                                                                        host(s)                    description\n     0       0     (t09,t10,pt1h) (xxxx-wxx-1,xxxx-wxx-5,p4d) the tmo click                                                                       jamie colby (m,t), harris faulkner (w\u2013f)                current events.\n     1       1 (t10,t10:30,pt30m) (xxxx-wxx-1,xxxx-wxx-5,p4d)          none lauren green (m), uma pemmaraju (t), gregg jarrett (w), arthel neville (f), rick folbaum (f), heather childers                current events.\n     2       2 (t11,t11:30,pt30m) (xxxx-wxx-1,xxxx-wxx-5,p4d)      fbn live                                                                                               lauren simonetti business news and information.\n*/"
    },
    "db_path": "./research/tmp/2838e14c-9b66-41b8-9fda-e8141a39617c.db",
    "pred_text": [
        7
    ],
    "pred_sql": " SELECT COUNT(*) FROM w WHERE \n            {{\n                LLM('How many hosts per day?', 'w::host(s)')\n            }} = 1",
    "pred_has_ingredient": true,
    "example_map_outputs": [
        [
            "2",
            "6",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1"
        ]
    ]
    }
```
### <a id="good_example_4"></a> Good Example 4
```json 
{
        "id": "nt-40",
        "question": "what was the number of silver medals won by ukraine?",
        "table_id": "csv/203-csv/175.csv",
        "table": {
            "page_title": "Sailing at the 2004 Summer Olympics",
            "header": [
                "Rank",
                "Nation",
                "Gold",
                "Silver",
                "Bronze",
                "Total"
            ],
            "rows": [
                [
                    "1",
                    "Great Britain\u00a0(GBR)",
                    "2",
                    "1",
                    "2",
                    "5"
                ],
                [
                    "2",
                    "Brazil\u00a0(BRA)",
                    "2",
                    "0",
                    "0",
                    "2"
                ],
                [
                    "3",
                    "Spain\u00a0(ESP)",
                    "1",
                    "2",
                    "0",
                    "3"
                ],
                [
                    "4",
                    "Austria\u00a0(AUT)",
                    "1",
                    "1",
                    "0",
                    "2"
                ],
                [
                    "4",
                    "Greece\u00a0(GRE)",
                    "1",
                    "1",
                    "0",
                    "2"
                ],
                [
                    "4",
                    "United States\u00a0(USA)",
                    "1",
                    "1",
                    "0",
                    "2"
                ],
                [
                    "7",
                    "France\u00a0(FRA)",
                    "1",
                    "0",
                    "1",
                    "2"
                ],
                [
                    "7",
                    "Israel\u00a0(ISR)",
                    "1",
                    "0",
                    "0",
                    "1"
                ],
                [
                    "7",
                    "Norway\u00a0(NOR)",
                    "1",
                    "0",
                    "0",
                    "1"
                ],
                [
                    "10",
                    "Ukraine\u00a0(UKR)",
                    "0",
                    "2",
                    "0",
                    "2"
                ],
                [
                    "11",
                    "China\u00a0(CHN)",
                    "0",
                    "1",
                    "0",
                    "1"
                ],
                [
                    "11",
                    "Czech Republic\u00a0(CZE)",
                    "0",
                    "1",
                    "0",
                    "1"
                ],
                [
                    "11",
                    "Canada\u00a0(CAN)",
                    "0",
                    "1",
                    "0",
                    "1"
                ],
                [
                    "14",
                    "Denmark\u00a0(DEN)",
                    "0",
                    "0",
                    "2",
                    "2"
                ],
                [
                    "15",
                    "Argentina\u00a0(ARG)",
                    "0",
                    "0",
                    "1",
                    "1"
                ],
                [
                    "15",
                    "Italy\u00a0(ITA)",
                    "0",
                    "0",
                    "1",
                    "1"
                ],
                [
                    "15",
                    "Japan\u00a0(JPN)",
                    "0",
                    "0",
                    "1",
                    "1"
                ],
                [
                    "15",
                    "Poland\u00a0(POL)",
                    "0",
                    "0",
                    "1",
                    "1"
                ],
                [
                    "15",
                    "Slovenia\u00a0(SLO)",
                    "0",
                    "0",
                    "1",
                    "1"
                ],
                [
                    "15",
                    "Sweden\u00a0(SWE)",
                    "0",
                    "0",
                    "1",
                    "1"
                ],
                [
                    "Total",
                    "",
                    "11",
                    "11",
                    "11",
                    "33"
                ]
            ]
        },
        "sql": "SELECT `Silver` FROM w WHERE `Nation` = 'ukraine'",
        "answer_text": [
            "2"
        ],
        "input_program_args": {
            "bridge_hints": "",
            "endpoint_name": "gpt-4",
            "examples": [
                {
                    "blendsql": "SELECT DISTINCT description, merchant FROM transactions WHERE\n    {{LLM('is this a pizza shop?', 'transactions::merchant', endpoint_name='gpt-4')}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'transactions::description',\n           binary=0,\n           example_outputs='TX;CA;MA;-1;',\n           endpoint_name='gpt-4'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                    "question": "Pizza shops in MA",
                    "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
                },
                {
                    "blendsql": "SELECT DISTINCT description, merchant, child_category, date FROM transactions WHERE\n      {{LLM('would this gift be appropriate for a child?', 'transactions::description', enpoint_name='gpt-4')}} = 1\n      AND child_category = 'Gifts'",
                    "question": "Show gifts for my kid",
                    "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
                }
            ],
            "program": "\n{{#system~}}\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{~/system}}\n\n{{#user~}}\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL:\n{{~/user}}\n\n{{#assistant~}}\n{{gen \"result\" temperature=0.0}}\n{{~/assistant}}\n",
            "question": "what was the number of silver medals won by ukraine?",
            "serialized_db": "CREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"rank\" TEXT,\n  \"nation\" TEXT,\n  \"gold\" INTEGER,\n  \"silver\" INTEGER,\n  \"bronze\" INTEGER,\n  \"total\" INTEGER\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id rank              nation  gold  silver  bronze  total\n     0       0    1 great britain\u00a0(gbr)     2       1       2      5\n     1       1    2        brazil\u00a0(bra)     2       0       0      2\n     2       2    3         spain\u00a0(esp)     1       2       0      3\n*/"
        },
        "db_path": "./research/tmp/66b50e28-4ffd-4890-8236-5a7f842f284b.db",
        "pred_text": [
            2
        ],
        "pred_sql": "SELECT silver FROM w WHERE {{LLM('is this the country ukraine?', 'w::nation', endpoint_name='gpt-4')}} = 1"
    }
```

### <a id="good_example_5"></a> Good Example 5
```json
{
    "id": "nt-353",
    "question": "between bjorn ferry, simon elder and erik lesser - who had the most penalties?",
    "table_id": "csv/204-csv/664.csv",
    "table": {
        "page_title": "Biathlon World Championships 2013 \u2013 Men's pursuit",
        "header": [
            "Rank",
            "Bib",
            "Name",
            "Nationality",
            "Start",
            "Penalties (P+P+S+S)",
            "Time",
            "Deficit"
        ],
        "rows": [
            [
                "",
                "1",
                "Emil Hegle Svendsen",
                "Norway",
                "0:00",
                "1 (0+0+0+1)",
                "32:35.5",
                ""
            ],
            [
                "",
                "2",
                "Martin Fourcade",
                "France",
                "0:08",
                "2 (0+1+1+0)",
                "32:35.6",
                "+0.1"
            ],
            [
                "",
                "7",
                "Anton Shipulin",
                "Russia",
                "0:33",
                "1 (0+0+1+0)",
                "32:39.1",
                "+3.6"
            ],
            [
                "4",
                "5",
                "Dmitry Malyshko",
                "Russia",
                "0:23",
                "0 (0+0+0+0)",
                "32:40.9",
                "+5.4"
            ],
            [
                "5",
                "15",
                "Dominik Landertinger",
                "Austria",
                "0:58",
                "0 (0+0+0+0)",
                "32:55.9",
                "+20.4"
            ],
            [
                "6",
                "3",
                "Jakov Fak",
                "Slovenia",
                "0:11",
                "1 (0+0+0+1)",
                "33:05.2",
                "+29.7"
            ],
            [
                "7",
                "8",
                "Fredrik Lindstr\u00f6m",
                "Sweden",
                "0:38",
                "1 (0+0+1+0)",
                "33:18.9",
                "+43.4"
            ],
            [
                "8",
                "6",
                "Alexis B\u0153uf",
                "France",
                "0:25",
                "2 (0+0+2+0)",
                "33:22.1",
                "+46.6"
            ],
            [
                "9",
                "13",
                "Bj\u00f6rn Ferry",
                "Sweden",
                "0:52",
                "0 (0+0+0+0)",
                "33:23.6",
                "+48.1"
            ],
            [
                "10",
                "4",
                "Ole Einar Bj\u00f8rndalen",
                "Norway",
                "0:20",
                "4 (2+0+1+1)",
                "33:27.0",
                "+51.5"
            ],
            [
                "11",
                "14",
                "Lukas Hofer",
                "Italy",
                "0:57",
                "2 (0+2+0+0)",
                "33:28.8",
                "+53.3"
            ],
            [
                "12",
                "10",
                "Simon Eder",
                "Austria",
                "0:39",
                "3 (0+0+2+1)",
                "33:50.5",
                "+1:15.0"
            ],
            [
                "13",
                "32",
                "Lowell Bailey",
                "United States",
                "1:38",
                "0 (0+0+0+0)",
                "33:56.4",
                "+1:20.9"
            ],
            [
                "14",
                "12",
                "Erik Lesser",
                "Germany",
                "0:49",
                "4 (1+0+2+1)",
                "34:06.0",
                "+1:30.5"
            ],
            [
                "15",
                "36",
                "Jean-Philippe Leguellec",
                "Canada",
                "1:45",
                "1 (0+0+0+1)",
                "34:08.7",
                "+1:33.2"
            ],
            [
                "16",
                "9",
                "Evgeny Ustyugov",
                "Russia",
                "0:38",
                "3 (0+1+2+0)",
                "34:16.7",
                "+1:41.2"
            ],
            [
                "17",
                "18",
                "Tarjei B\u00f8",
                "Norway",
                "1:14",
                "3 (2+0+1+0)",
                "34:18.9",
                "+1:43.4"
            ],
            [
                "18",
                "29",
                "Simon Schempp",
                "Germany",
                "1:36",
                "1 (0+0+0+1)",
                "34:24.4",
                "+1:48.9"
            ],
            [
                "19",
                "11",
                "Krasimir Anev",
                "Bulgaria",
                "0:48",
                "2 (0+1+0+1)",
                "34:24.7",
                "+1:49.2"
            ],
            [
                "20",
                "37",
                "Henrik L'Abee-Lund",
                "Norway",
                "1:45",
                "1 (0+0+0+1)",
                "34:36.1",
                "+2:00.6"
            ],
            [
                "21",
                "16",
                "Arnd Peiffer",
                "Germany",
                "1:08",
                "3 (0+0+2+1)",
                "34:39.8",
                "+2:04.3"
            ],
            [
                "22",
                "23",
                "Andreas Birnbacher",
                "Germany",
                "1:25",
                "3 (0+1+1+1)",
                "34:41.7",
                "+2:06.2"
            ],
            [
                "23",
                "34",
                "Simon Fourcade",
                "France",
                "1:42",
                "3 (0+3+0+0)",
                "34:51.1",
                "+2:15.6"
            ],
            [
                "24",
                "21",
                "Klemen Bauer",
                "Slovenia",
                "1:16",
                "4 (1+0+3+0)",
                "34:55.4",
                "+2:19.9"
            ],
            [
                "25",
                "26",
                "Christian De Lorenzi",
                "Italy",
                "1:30",
                "1 (0+0+0+1)",
                "34:57.5",
                "+2:22.0"
            ],
            [
                "26",
                "20",
                "Jean-Guillaume B\u00e9atrix",
                "France",
                "1:15",
                "4 (0+1+2+1)",
                "34:58.4",
                "+2:22.9"
            ],
            [
                "27",
                "33",
                "Daniel Mesotitsch",
                "Austria",
                "1:39",
                "3 (0+0+2+1)",
                "34:58.7",
                "+2:23.2"
            ],
            [
                "28",
                "19",
                "Evgeniy Garanichev",
                "Russia",
                "1:14",
                "5 (2+0+2+1)",
                "35:04.2",
                "+2:28.7"
            ],
            [
                "29",
                "30",
                "Matej Kaz\u00e1r",
                "Slovakia",
                "1:36",
                "5 (2+1+1+1)",
                "35:04.9",
                "+2:29.4"
            ],
            [
                "30",
                "17",
                "Andriy Deryzemlya",
                "Ukraine",
                "1:13",
                "4 (0+0+2+2)",
                "35:05.3",
                "+2:29.8"
            ],
            [
                "31",
                "35",
                "Ond\u0159ej Moravec",
                "Czech Republic",
                "1:42",
                "3 (0+1+1+1)",
                "35:05.3",
                "+2:48.2"
            ],
            [
                "32",
                "28",
                "Tim Burke",
                "United States",
                "1:36",
                "4 (0+1+2+1)",
                "35:33.8",
                "+2:58.3"
            ],
            [
                "33",
                "58",
                "Dominik Windisch",
                "Italy",
                "2:31",
                "2 (0+0+1+1)",
                "35:36.4",
                "+3:00.9"
            ],
            [
                "34",
                "54",
                "Carl Johan Bergman",
                "Sweden",
                "2:19",
                "2 (0+0+1+1)",
                "35:37.0",
                "+3:01.5"
            ],
            [
                "35",
                "25",
                "Benjamin Weger",
                "Switzerland",
                "1:28",
                "3 (0+1+1+1)",
                "36:13.1",
                "+3:37.6"
            ],
            [
                "36",
                "42",
                "Artem Pryma",
                "Ukraine",
                "1:59",
                "3 (1+2+0+0)",
                "36:13.2",
                "+3:37.7"
            ],
            [
                "37",
                "50",
                "Pavol Hurajt",
                "Slovakia",
                "2:14",
                "2 (0+1+0+1)",
                "36:18.9",
                "+3:43.4"
            ],
            [
                "38",
                "22",
                "Tomas Kauk\u0117nas",
                "Lithuania",
                "1:20",
                "4 (0+0+2+2)",
                "36:33.4",
                "+3:57.9"
            ],
            [
                "39",
                "43",
                "Vladimir Iliev",
                "Bulgaria",
                "2:03",
                "5 (1+1+2+1)",
                "36:43.4",
                "+4:07.9"
            ],
            [
                "40",
                "40",
                "Edgars Piksons",
                "Latvia",
                "1:55",
                "4 (1+0+2+1)",
                "36:49.8",
                "+4:14.3"
            ],
            [
                "41",
                "60",
                "Christian Martinelli",
                "Italy",
                "2:36",
                "2 (1+0+0+1)",
                "36:53.0",
                "+4:17.5"
            ],
            [
                "42",
                "27",
                "Jaroslav Soukup",
                "Czech Republic",
                "1:32",
                "6 (2+2+2+0)",
                "36:59.8",
                "+4:24.3"
            ],
            [
                "43",
                "53",
                "Leif Nordgren",
                "United States",
                "2:19",
                "6 (2+2+2+0)",
                "37:08.3",
                "+4:32.8"
            ],
            [
                "44",
                "56",
                "Hidenori Isa",
                "Japan",
                "2:27",
                "2 (0+0+0+2)",
                "37:13.9",
                "+4:38.4"
            ],
            [
                "45",
                "41",
                "Serhiy Semenov",
                "Ukraine",
                "1:55",
                "5 (3+1+1+0)",
                "37:14.0",
                "+4:38.5"
            ],
            [
                "46",
                "24",
                "Zden\u011bk V\u00edtek",
                "Czech Republic",
                "1:25",
                "7 (2+2+1+2)",
                "37:21.0",
                "+4:45.5"
            ],
            [
                "47",
                "49",
                "Milanko Petrovi\u0107",
                "Serbia",
                "2:14",
                "6 (2+2+2+0)",
                "37:26.6",
                "+4:51.1"
            ],
            [
                "48",
                "31",
                "Kauri K\u00f5iv",
                "Estonia",
                "1:37",
                "6 (2+1+2+1)",
                "37:29.9",
                "+4:54.4"
            ],
            [
                "49",
                "47",
                "Yan Savitskiy",
                "Kazakhstan",
                "2:11",
                "5 (2+2+0+1)",
                "37:51.8",
                "+5:16.3"
            ],
            [
                "50",
                "45",
                "Scott Gow",
                "Canada",
                "2:06",
                "5 (0+0+1+4)",
                "37:57.4",
                "+5:21.9"
            ],
            [
                "51",
                "48",
                "Claudio B\u00f6ckli",
                "Switzerland",
                "2:13",
                "3 (2+0+0+1)",
                "38:02.8",
                "+5:27.3"
            ],
            [
                "52",
                "44",
                "Russell Currier",
                "United States",
                "2:05",
                "7 (3+1+2+1)",
                "38:06.2",
                "+5:30.7"
            ],
            [
                "53",
                "57",
                "Vladimir Chepelin",
                "Belarus",
                "2:28",
                "4 (0+1+2+1)",
                "38:06.5",
                "+5:31.0"
            ],
            [
                "54",
                "46",
                "Aliaksandr Babchyn",
                "Belarus",
                "2:06",
                "6 (3+2+1+0)",
                "38:19.8",
                "+5:44.3"
            ],
            [
                "55",
                "59",
                "Miroslav Matia\u0161ko",
                "Slovakia",
                "2:36",
                "6 (3+0+2+1)",
                "38:54.6",
                "+6:19.1"
            ],
            [
                "56",
                "38",
                "Jarkko Kauppinen",
                "Finland",
                "1:46",
                "6 (1+1+3+1)",
                "39:06.6",
                "+6:31.1"
            ],
            [
                "57",
                "51",
                "Junji Nagai",
                "Japan",
                "2:17",
                "7 (0+2+3+2)",
                "39:49.9",
                "+7:14.4"
            ],
            [
                "",
                "55",
                "Stefan Gavrila",
                "Romania",
                "2:25",
                "10 (4+3+3)",
                "LAP",
                ""
            ],
            [
                "",
                "39",
                "Sergei Sednev",
                "Ukraine",
                "1:54",
                "",
                "DNS",
                ""
            ],
            [
                "",
                "52",
                "Michal \u0160lesingr",
                "Czech Republic",
                "2:18",
                "",
                "DNS",
                ""
            ]
        ]
    },
    "sql": "SELECT `Name` FROM w WHERE `Name` IN ( 'bjorn ferry' , 'simon eder' , 'erik lesser' ) ORDER by `Penalties (P+P+S+S)` desc LIMIT 1",
    "answer_text": [
        "Erik Lesser"
    ],
    "input_program_args": {
        "bridge_hints": "name ( erik lesser )",
        "examples": [
            {
                "blendsql": "SELECT DISTINCT description, merchant FROM transactions WHERE\n    {{LLM('is this a pizza shop?', 'transactions::merchant', endpoint_name='gpt-4')}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'transactions::description',\n           binary=0,\n           example_outputs='TX;CA;MA;-1;'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                "question": "Pizza shops in MA",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            },
            {
                "blendsql": "SELECT DISTINCT description, merchant, child_category, date FROM transactions WHERE\n      {{LLM('would this gift be appropriate for a child?', 'transactions::description')}} = 1\n      AND child_category = 'Gifts'",
                "question": "Show gifts for my kid",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            }
        ],
        "program": "\n{{#system~}}\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{~/system}}\n\n{{#user~}}\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL:\n{{~/user}}\n\n{{#assistant~}}\n{{gen \"result\" temperature=0.0}}\n{{~/assistant}}\n",
        "question": "between bjorn ferry, simon elder and erik lesser - who had the most penalties?",
        "serialized_db": "CREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"rank\" REAL,\n  \"bib\" INTEGER,\n  \"name\" TEXT,\n  \"nationality\" TEXT,\n  \"start\" TEXT,\n  \"penalties (p+p+s+s)\" TEXT,\n  \"time\" TEXT,\n  \"deficit\" TEXT\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id rank  bib                name nationality start penalties (p+p+s+s)    time deficit\n     0       0 None    1 emil hegle svendsen      norway   0:0         1 (0+0+0+1) 32:35.5    none\n     1       1 None    2     martin fourcade      france   0:8         2 (0+1+1+0) 32:35.6    +0.1\n     2       2 None    7      anton shipulin      russia  0:33         1 (0+0+1+0) 32:39.1    +3.6\n*/"
    },
    "db_path": "./research/tmp/bc195c10-5a92-4b4c-a665-59548a2a822a.db",
    "pred_text": [
        "erik lesser",
        null
    ],
    "pred_sql": "SELECT name, MAX({{LLM('how many penalties does this person have?', 'w::penalties (p+p+s+s)', binary=0, example_outputs='0;1;2;3;4;5;6;7;8;9;10;') }}) as penalties FROM w WHERE name IN ('bjorn ferry', 'simon elder', 'erik lesser') GROUP BY name ORDER BY penalties DESC LIMIT 1"
}
```
## Interesting Mistakes
1) Correctly checks out the 'remarks' column, but should be 'how many pylons does this remark mention?'
```json
{
        "id": "nt-53",
        "question": "what is the total number of pylons listed?",
        "table_id": "csv/203-csv/375.csv",
        "table": {
            "page_title": "Aerial lift pylon",
            "header": [
                "Name",
                "Year built",
                "Country",
                "Town",
                "Height",
                "Remarks"
            ],
            "rows": [
                [
                    "Glacial Aerial Tramway Kaprun III",
                    "1966",
                    "Austria",
                    "Kaprun",
                    "113.6 m",
                    "The tallest pylon is on the third section"
                ],
                [
                    "Mississippi Aerial River Transit",
                    "1984",
                    "USA",
                    "New Orleans",
                    "109 m",
                    "The tallest pylon on a gondola lift; on 87 m pile foundations; demolished 1994"
                ],
                [
                    "Torre Jaume I",
                    "1931",
                    "Spain",
                    "Barcelona",
                    "107 m",
                    "Intermediate stop of the harbour aerial tramway, also observation tower"
                ],
                [
                    "Gant Hohtaelli aerial tramway",
                    "",
                    "Switzerland",
                    "Zermatt",
                    "94 m",
                    "One pylon"
                ],
                [
                    "London Emirates Air Line",
                    "Early 2012",
                    "UK",
                    "London",
                    "88 m",
                    "North Main Tower. River Thames aerial crossing between Greenwich Peninsular and Royal Docks"
                ],
                [
                    "Singapore cable car",
                    "1972",
                    "Singapore",
                    "Singapore",
                    "88 m",
                    "Pylon I"
                ],
                [
                    "Eibsee Aerial Tramway",
                    "1962",
                    "Germany",
                    "Garmisch-Partenkirchen",
                    "85 m",
                    "Pylon II"
                ],
                [
                    "Nizhny Novgorod Volga Aerial Tramway, Tower 4 & 5",
                    "2012",
                    "Russia",
                    "Nizhny Novgorod",
                    "82 m",
                    ""
                ],
                [
                    "Mittersill goods aerial tramway",
                    "194?",
                    "Austria",
                    "Mittersill",
                    "80 m",
                    "Two pylons for a tramway that never went in service and was demolished in the 1950s. One of the pylons was built of timber, the other of steel."
                ],
                [
                    "Singapore cable car",
                    "1972",
                    "Singapore",
                    "Singapore",
                    "80 m",
                    "Pylon II"
                ],
                [
                    "3S Aerial Tramway",
                    "2004",
                    "Austria",
                    "Kitzb\u00fchel",
                    "80 m",
                    "One pylon"
                ],
                [
                    "Torre Sant Sebastia",
                    "1931",
                    "Spain",
                    "Barcelona",
                    "78 m",
                    "Terminal of harbour aerial tramway"
                ],
                [
                    "Roosevelt Island Tramway",
                    "1976",
                    "USA",
                    "New York City",
                    "76 m",
                    "Central pylon of commuter tramway"
                ],
                [
                    "Wendelstein Aerial Tramway",
                    "1970",
                    "Germany",
                    "Bayerischzell-Osterhofen",
                    "75 m",
                    ""
                ],
                [
                    "Vinpearl Cable Car",
                    "2007",
                    "Vietnam",
                    "Nha Trang",
                    "75 m",
                    "7 pylons standing in the sea. Total height from sea bed is 115 m"
                ],
                [
                    "Sandia Peak Tramway",
                    "1965",
                    "USA",
                    "Albuquerque",
                    "70.7 m",
                    "Pylon 1, inclined at an angle of 18 degrees"
                ],
                [
                    "Eibsee Aerial Tramway",
                    "1962",
                    "Germany",
                    "Garmisch-Partenkirchen",
                    "65 m",
                    "Pylon I"
                ]
            ]
        },
        "sql": "SELECT COUNT ( `Name` ) FROM w",
        "answer_text": [
            "17"
        ],
        "input_program_args": {
            "bridge_hints": "",
            "examples": [
                {
                    "blendsql": "SELECT DISTINCT description, merchant FROM transactions WHERE\n    {{LLM('is this a pizza shop?', 'transactions::merchant', endpoint_name='gpt-4')}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'transactions::description',\n           binary=0,\n           example_outputs='TX;CA;MA;-1;'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                    "question": "Pizza shops in MA",
                    "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
                },
                {
                    "blendsql": "SELECT DISTINCT description, merchant, child_category, date FROM transactions WHERE\n      {{LLM('would this gift be appropriate for a child?', 'transactions::description')}} = 1\n      AND child_category = 'Gifts'",
                    "question": "Show gifts for my kid",
                    "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
                }
            ],
            "program": "\n{{#system~}}\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{~/system}}\n\n{{#user~}}\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL:\n{{~/user}}\n\n{{#assistant~}}\n{{gen \"result\" temperature=0.0}}\n{{~/assistant}}\n",
            "question": "what is the total number of pylons listed?",
            "serialized_db": "CREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"name\" TEXT,\n  \"year built\" TEXT,\n  \"country\" TEXT,\n  \"town\" TEXT,\n  \"height\" TEXT,\n  \"remarks\" TEXT\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id                              name year built country        town  height                                                                        remarks\n     0       0 glacial aerial tramway kaprun iii       1966 austria      kaprun 113.6 m                                      the tallest pylon is on the third section\n     1       1  mississippi aerial river transit       1984     usa new orleans   109 m the tallest pylon on a gondola lift; on 87 m pile foundations; demolished 1994\n     2       2                     torre jaume i       1931   spain   barcelona   107 m        intermediate stop of the harbour aerial tramway, also observation tower\n*/"
        },
        "db_path": "./research/tmp/ec0862aa-6fa1-4032-acb0-e340eba39301.db",
        "pred_text": [
            12
        ],
        "pred_sql": "SELECT COUNT(*) FROM w WHERE {{LLM('does this remark mention a pylon?', 'w::remarks')}} = 1"
    }
```

2) Failing to add arguments `binary=0`
```json
{
    "id": "nt-395",
    "question": "what is the pylon with the least height listed here?",
    "table_id": "csv/203-csv/375.csv",
    "table": {
        "page_title": "Aerial lift pylon",
        "header": [
            "Name",
            "Year built",
            "Country",
            "Town",
            "Height",
            "Remarks"
        ],
        "rows": [
            [
                "Glacial Aerial Tramway Kaprun III",
                "1966",
                "Austria",
                "Kaprun",
                "113.6 m",
                "The tallest pylon is on the third section"
            ],
            [
                "Mississippi Aerial River Transit",
                "1984",
                "USA",
                "New Orleans",
                "109 m",
                "The tallest pylon on a gondola lift; on 87 m pile foundations; demolished 1994"
            ],
            [
                "Torre Jaume I",
                "1931",
                "Spain",
                "Barcelona",
                "107 m",
                "Intermediate stop of the harbour aerial tramway, also observation tower"
            ],
            [
                "Gant Hohtaelli aerial tramway",
                "",
                "Switzerland",
                "Zermatt",
                "94 m",
                "One pylon"
            ],
            [
                "London Emirates Air Line",
                "Early 2012",
                "UK",
                "London",
                "88 m",
                "North Main Tower. River Thames aerial crossing between Greenwich Peninsular and Royal Docks"
            ],
            [
                "Singapore cable car",
                "1972",
                "Singapore",
                "Singapore",
                "88 m",
                "Pylon I"
            ],
            [
                "Eibsee Aerial Tramway",
                "1962",
                "Germany",
                "Garmisch-Partenkirchen",
                "85 m",
                "Pylon II"
            ],
            [
                "Nizhny Novgorod Volga Aerial Tramway, Tower 4 & 5",
                "2012",
                "Russia",
                "Nizhny Novgorod",
                "82 m",
                ""
            ],
            [
                "Mittersill goods aerial tramway",
                "194?",
                "Austria",
                "Mittersill",
                "80 m",
                "Two pylons for a tramway that never went in service and was demolished in the 1950s. One of the pylons was built of timber, the other of steel."
            ],
            [
                "Singapore cable car",
                "1972",
                "Singapore",
                "Singapore",
                "80 m",
                "Pylon II"
            ],
            [
                "3S Aerial Tramway",
                "2004",
                "Austria",
                "Kitzb\u00fchel",
                "80 m",
                "One pylon"
            ],
            [
                "Torre Sant Sebastia",
                "1931",
                "Spain",
                "Barcelona",
                "78 m",
                "Terminal of harbour aerial tramway"
            ],
            [
                "Roosevelt Island Tramway",
                "1976",
                "USA",
                "New York City",
                "76 m",
                "Central pylon of commuter tramway"
            ],
            [
                "Wendelstein Aerial Tramway",
                "1970",
                "Germany",
                "Bayerischzell-Osterhofen",
                "75 m",
                ""
            ],
            [
                "Vinpearl Cable Car",
                "2007",
                "Vietnam",
                "Nha Trang",
                "75 m",
                "7 pylons standing in the sea. Total height from sea bed is 115 m"
            ],
            [
                "Sandia Peak Tramway",
                "1965",
                "USA",
                "Albuquerque",
                "70.7 m",
                "Pylon 1, inclined at an angle of 18 degrees"
            ],
            [
                "Eibsee Aerial Tramway",
                "1962",
                "Germany",
                "Garmisch-Partenkirchen",
                "65 m",
                "Pylon I"
            ]
        ]
    },
    "sql": "SELECT `Name` FROM w ORDER by `Height` asc LIMIT 1",
    "answer_text": [
        "Eibsee Aerial Tramway"
    ],
    "input_program_args": {
        "bridge_hints": "",
        "examples": [
            {
                "blendsql": "SELECT DISTINCT description, merchant FROM transactions WHERE\n    {{LLM('is this a pizza shop?', 'transactions::merchant', endpoint_name='gpt-4')}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'transactions::description',\n           binary=0,\n           example_outputs='TX;CA;MA;-1;'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                "question": "Pizza shops in MA",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            },
            {
                "blendsql": "SELECT DISTINCT description, merchant, child_category, date FROM transactions WHERE\n      {{LLM('would this gift be appropriate for a child?', 'transactions::description')}} = 1\n      AND child_category = 'Gifts'",
                "question": "Show gifts for my kid",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            }
        ],
        "program": "\n{{#system~}}\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{~/system}}\n\n{{#user~}}\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL:\n{{~/user}}\n\n{{#assistant~}}\n{{gen \"result\" temperature=0.0}}\n{{~/assistant}}\n",
        "question": "what is the pylon with the least height listed here?",
        "serialized_db": "CREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"name\" TEXT,\n  \"year built\" TEXT,\n  \"country\" TEXT,\n  \"town\" TEXT,\n  \"height\" TEXT,\n  \"remarks\" TEXT\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id                              name year built country        town  height                                                                        remarks\n     0       0 glacial aerial tramway kaprun iii       1966 austria      kaprun 113.6 m                                      the tallest pylon is on the third section\n     1       1  mississippi aerial river transit       1984     usa new orleans   109 m the tallest pylon on a gondola lift; on 87 m pile foundations; demolished 1994\n     2       2                     torre jaume i       1931   spain   barcelona   107 m        intermediate stop of the harbour aerial tramway, also observation tower\n*/"
    },
    "db_path": "./research/tmp/4820622f-ef25-43f9-aa93-6501cd7db235.db",
    "pred_text": [
        "mississippi aerial river transit",
        "109 m"
    ],
    "pred_sql": "SELECT name, height FROM w ORDER BY {{LLM('what is the numerical value of this height?', 'w::height')}} ASC LIMIT 1"
}
```

3) Could have reused previous ingredient (maybe bad few-shot prompts at fault here)
```json
{
    "id": "nt-397",
    "question": "how many provinces have the same number of districts as kabul?",
    "table_id": "csv/204-csv/704.csv",
    "table": {
        "page_title": "Outline of Afghanistan",
        "header": [
            "Province",
            "Map #",
            "ISO 3166-2:AF",
            "Centers",
            "Population",
            "Area (km\u00b2)",
            "Language",
            "Notes",
            "U.N. Region"
        ],
        "rows": [
            [
                "Badakhshan",
                "30",
                "AF-BDS",
                "Fayzabad",
                "823,000",
                "44,059",
                "Dari Persian,",
                "Composed of 29 districts",
                "North East Afghanistan"
            ],
            [
                "Badghis",
                "4",
                "AF-BDG",
                "Qala i Naw",
                "429,500",
                "20,591",
                "Dari Persian, Pashto",
                "7 districts",
                "West Afghanistan"
            ],
            [
                "Baghlan",
                "19",
                "AF-BGL",
                "Puli Khumri",
                "779,000",
                "21,118",
                "Dari Persian Uzbek, Turkmen, Pashto",
                "16 districts",
                "North East Afghanistan"
            ],
            [
                "Balkh",
                "13",
                "AF-BAL",
                "Mazari Sharif",
                "1,096,100",
                "17,249",
                "Dari Persian,",
                "15 districts",
                "North Afghanistan"
            ],
            [
                "Bamyan",
                "15",
                "AF-BAM",
                "Bamyan",
                "387,300",
                "14,175",
                "Dari Persian",
                "7 districts",
                "East Central Afghanistan"
            ],
            [
                "Daykundi",
                "10",
                "AF-DAY",
                "Nili",
                "399,600",
                "8,088",
                "Dari Persian",
                "8 districts\\nFormed from Or\u016bzg\u0101n Province in 2004",
                "South West Afghanistan"
            ],
            [
                "Farah",
                "2",
                "AF-FRA",
                "Farah",
                "438,000",
                "48,471",
                "Pashto, Dari Persian, Baloch",
                "11 districts",
                "West Afghanistan"
            ],
            [
                "Faryab",
                "5",
                "AF-FYB",
                "Maymana",
                "858,600",
                "20,293",
                "Dari Persian, Turkmen and Uzbek",
                "14 districts",
                "West Afghanistan"
            ],
            [
                "Ghazni",
                "16",
                "AF-GHA",
                "Ghazni",
                "931,000",
                "22,915",
                "Pashto, Dari Persian",
                "19 districts",
                "South Afghanistan"
            ],
            [
                "Ghor",
                "6",
                "AF-GHO",
                "Chaghcharan",
                "485,000",
                "36,479",
                "Dari Persian",
                "10 districts",
                "East Central Afghanistan"
            ],
            [
                "Helmand",
                "7",
                "AF-HEL",
                "Lashkar Gah",
                "745,000",
                "58,584",
                "Pashto,Baloch",
                "13 districts",
                "South West Afghanistan"
            ],
            [
                "Herat",
                "1",
                "AF-HER",
                "Herat",
                "1,182,000",
                "54,778",
                "Dari Persian, Pashto",
                "15 districts",
                "West Afghanistan"
            ],
            [
                "Jowzjan",
                "8",
                "AF-JOW",
                "Sheberghan",
                "441,000",
                "11,798",
                "Uzbek, Dari Persian",
                "9 districts",
                "North Afghanistan"
            ],
            [
                "Kabul",
                "22",
                "AF-KAB",
                "Kabul",
                "3,314,000",
                "4,462",
                "Pashto, Dari Persian, Turkmen, Baloch and Uzbek",
                "15 districts",
                "Central Afghanistan"
            ],
            [
                "Kandahar",
                "12",
                "AF-KAN",
                "Kandahar",
                "886,000",
                "54,022",
                "Pashto, Dari Persian,Baloch",
                "16 districts",
                "South West Afghanistan"
            ],
            [
                "Kapisa",
                "29",
                "AF-KAP",
                "Mahmud-i-Raqi",
                "360,000",
                "1,842",
                "Dari Persian",
                "7 districts",
                "Central Afghanistan"
            ],
            [
                "Khost",
                "26",
                "AF-KHO",
                "Khost",
                "300,000",
                "4,152",
                "Pashto",
                "13 districts",
                "South Afghanistan"
            ],
            [
                "Kunar",
                "34",
                "AF-KNR",
                "Asadabad",
                "321,000",
                "4,942",
                "Pashto",
                "15 districts",
                "East Afghanistan"
            ],
            [
                "Kunduz",
                "18",
                "AF-KDZ",
                "Kunduz",
                "820,000",
                "8,040",
                "Dari Persian, Uzbek, Turkmen, Pashto",
                "7 districts",
                "North East Afghanistan"
            ],
            [
                "Laghman",
                "32",
                "AF-LAG",
                "Mihtarlam District",
                "373,000",
                "3,843",
                "Pashto, Dari Persian and Pashai",
                "5 districts",
                "East Afghanistan"
            ],
            [
                "Lowgar",
                "23",
                "AF-LOW",
                "Pul-i-Alam",
                "292,000",
                "3,880",
                "Dari Persian, Pashto",
                "7 districts",
                "Central Afghanistan"
            ],
            [
                "Nangarhar",
                "33",
                "AF-NAN",
                "Jalalabad",
                "1,089,000",
                "7,727",
                "Pashto, Dari Persian",
                "23 districts",
                "East Afghanistan"
            ],
            [
                "Nimruz",
                "3",
                "AF-NIM",
                "Zaranj",
                "149,000",
                "41,005",
                "Balochi, Dari Persian and Balochi",
                "5 districts",
                "South West Afghanistan"
            ],
            [
                "Nurestan",
                "31",
                "AF-NUR",
                "Parun",
                "112,000",
                "9,225",
                "Nuristani, Pashto, Dari Persian",
                "7 districts",
                "East Afghanistan"
            ],
            [
                "Or\u016bzg\u0101n",
                "11",
                "AF-ORU",
                "Tarin Kowt",
                "627,000",
                "22,696",
                "Pashto",
                "6 districts",
                "South West Afghanistan"
            ],
            [
                "Paktia",
                "24",
                "AF-PIA",
                "Gardez",
                "415,000",
                "6,432",
                "Pashto, Dari Persian",
                "11 districts",
                "South Afghanistan"
            ],
            [
                "Paktika",
                "25",
                "AF-PKA",
                "Sharan",
                "352,000",
                "19,482",
                "Pashto",
                "15 districts",
                "South Afghanistan"
            ],
            [
                "Panjshir",
                "28",
                "AF-PAN",
                "Bazarak",
                "128,620",
                "3,610",
                "Dari Persian",
                "5 districts\\nCreated in 2004 from Parwan Province",
                "Central Afghanistan"
            ],
            [
                "Parwan",
                "20",
                "AF-PAR",
                "Charikar",
                "726,000",
                "5,974",
                "Dari Persian",
                "Composed of 9 districts",
                "Central Afghanistan"
            ],
            [
                "Samangan",
                "14",
                "AF-SAM",
                "Aybak",
                "378,000",
                "11,262",
                "Dari Persian, Uzbek",
                "5 districts",
                "North Afghanistan"
            ],
            [
                "Sar-e Pol",
                "9",
                "AF-SAR",
                "Sar-e Pol",
                "468,000",
                "15,999",
                "Dari Persian and Uzbek",
                "6 districts",
                "North Afghanistan"
            ],
            [
                "Takhar",
                "27",
                "AF-TAK",
                "Taloqan",
                "810,800",
                "12,333",
                "Dari Persian, and Uzbek",
                "12 districts",
                "North East Afghanistan"
            ],
            [
                "Wardak",
                "21",
                "AF-WAR",
                "Meydan Shahr",
                "413,000",
                "8,938",
                "Pashto, Dari Persian",
                "9 districts",
                "Central Afghanistan"
            ],
            [
                "Zabul",
                "17",
                "AF-ZAB",
                "Qalat",
                "365,920",
                "17,343",
                "Pashto",
                "9 districts",
                "South West Afghanistan"
            ]
        ]
    },
    "sql": "SELECT COUNT ( `Province` ) FROM w WHERE `Province` != 'kabul' and `Notes` = ( SELECT `Notes` FROM w WHERE `Province` = 'kabul' )",
    "answer_text": [
        "4"
    ],
    "input_program_args": {
        "bridge_hints": "province ( kabul )\ncenters ( kabul )",
        "examples": [
            {
                "blendsql": "SELECT DISTINCT description, merchant FROM transactions WHERE\n    {{LLM('is this a pizza shop?', 'transactions::merchant', endpoint_name='gpt-4')}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'transactions::description',\n           binary=0,\n           example_outputs='TX;CA;MA;-1;'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                "question": "Pizza shops in MA",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            },
            {
                "blendsql": "SELECT DISTINCT description, merchant, child_category, date FROM transactions WHERE\n      {{LLM('would this gift be appropriate for a child?', 'transactions::description')}} = 1\n      AND child_category = 'Gifts'",
                "question": "Show gifts for my kid",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            }
        ],
        "program": "\n{{#system~}}\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{~/system}}\n\n{{#user~}}\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL:\n{{~/user}}\n\n{{#assistant~}}\n{{gen \"result\" temperature=0.0}}\n{{~/assistant}}\n",
        "question": "how many provinces have the same number of districts as kabul?",
        "serialized_db": "CREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"province\" TEXT,\n  \"map #\" INTEGER,\n  \"iso 3166-2:af\" TEXT,\n  \"centers\" TEXT,\n  \"population\" INTEGER,\n  \"area (km\u00b2)\" INTEGER,\n  \"language\" TEXT,\n  \"notes\" TEXT,\n  \"u.n. region\" TEXT\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id   province  map # iso 3166-2:af     centers  population  area (km\u00b2)                            language                    notes            u.n. region\n     0       0 badakhshan     30        af-bds    fayzabad      823000       44059                       dari persian, composed of 29 districts north east afghanistan\n     1       1    badghis      4        af-bdg  qala i naw      429500       20591                dari persian, pashto              7 districts       west afghanistan\n     2       2    baghlan     19        af-bgl puli khumri      779000       21118 dari persian uzbek, turkmen, pashto             16 districts north east afghanistan\n*/"
    },
    "db_path": "./research/tmp/874c2ef9-1a93-49e3-99c9-4b7f79949c74.db",
    "pred_text": [
        34
    ],
    "pred_sql": "SELECT COUNT(DISTINCT province) FROM w WHERE\n    {{LLM('how many districts does this province have?', 'w::notes')}} = \n    {{LLM('how many districts does this province have?', 'w::notes', example_inputs='kabul')}}"
}
``` 
4) **IMPORTANT** Didn't know what 'that year' means if we don't pass in table header
```json
{
    "id": "nt-624",
    "question": "how many players made an appearance that year?",
    "table_id": "csv/203-csv/456.csv",
    "table": {
        "page_title": "1968 in Japanese football",
        "header": [
            "Player",
            "-1967",
            "03.30",
            "03.31",
            "04.04",
            "10.14",
            "1968",
            "Total"
        ],
        "rows": [
            [
                "Shigeo Yaegashi",
                "41(11)",
                "O",
                "O",
                "O",
                "O",
                "4(0)",
                "45(11)"
            ],
            [
                "Mitsuo Kamata",
                "37(2)",
                "O",
                "-",
                "O",
                "O",
                "3(0)",
                "40(2)"
            ],
            [
                "Masakatsu Miyamoto",
                "36(1)",
                "-",
                "O",
                "-",
                "O",
                "2(0)",
                "38(1)"
            ],
            [
                "Masashi Watanabe",
                "34(11)",
                "-",
                "O",
                "O",
                "-",
                "2(0)",
                "36(11)"
            ],
            [
                "Teruki Miyamoto",
                "33(15)",
                "O",
                "O",
                "O",
                "O",
                "4(0)",
                "37(15)"
            ],
            [
                "Ryuichi Sugiyama",
                "31(11)",
                "O",
                "O",
                "O(1)",
                "O",
                "4(1)",
                "35(12)"
            ],
            [
                "Hiroshi Katayama",
                "26(0)",
                "O",
                "-",
                "O",
                "O",
                "3(0)",
                "29(0)"
            ],
            [
                "Ryozo Suzuki",
                "22(0)",
                "O",
                "-",
                "O",
                "-",
                "2(0)",
                "24(0)"
            ],
            [
                "Kunishige Kamamoto",
                "17(21)",
                "O(2)",
                "O",
                "O(2)",
                "O(3)",
                "4(7)",
                "21(28)"
            ],
            [
                "Aritatsu Ogi",
                "16(5)",
                "O",
                "-",
                "O",
                "O",
                "3(0)",
                "19(5)"
            ],
            [
                "Yoshitada Yamaguchi",
                "16(0)",
                "O",
                "O",
                "O",
                "-",
                "3(0)",
                "19(0)"
            ],
            [
                "Kenzo Yokoyama",
                "16(0)",
                "O",
                "-",
                "O",
                "O",
                "3(0)",
                "19(0)"
            ],
            [
                "Hisao Kami",
                "13(0)",
                "O",
                "O",
                "-",
                "-",
                "2(0)",
                "15(0)"
            ],
            [
                "Takaji Mori",
                "9(1)",
                "O",
                "O",
                "O",
                "O",
                "4(0)",
                "13(1)"
            ],
            [
                "Ikuo Matsumoto",
                "7(1)",
                "O",
                "-",
                "-",
                "O",
                "2(0)",
                "9(1)"
            ],
            [
                "Yasuyuki Kuwahara",
                "5(3)",
                "-",
                "O(1)",
                "-",
                "O",
                "2(1)",
                "7(4)"
            ],
            [
                "Takeo Kimura",
                "3(1)",
                "O",
                "O",
                "-",
                "-",
                "2(0)",
                "5(1)"
            ],
            [
                "Kiyoshi Tomizawa",
                "3(0)",
                "-",
                "O",
                "-",
                "-",
                "1(0)",
                "4(0)"
            ],
            [
                "Koji Funamoto",
                "1(0)",
                "-",
                "O",
                "-",
                "-",
                "1(0)",
                "2(0)"
            ],
            [
                "Junji Kawano",
                "0(0)",
                "-",
                "O",
                "-",
                "-",
                "1(0)",
                "1(0)"
            ]
        ]
    },
    "sql": "SELECT COUNT ( `Player` ) FROM w",
    "answer_text": [
        "20"
    ],
    "input_program_args": {
        "bridge_hints": "",
        "examples": [
            {
                "blendsql": "SELECT DISTINCT description, merchant FROM transactions WHERE\n    {{LLM('is this a pizza shop?', 'transactions::merchant', endpoint_name='gpt-4')}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'transactions::description',\n           binary=0,\n           example_outputs='TX;CA;MA;-1;'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                "question": "Pizza shops in MA",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            },
            {
                "blendsql": "SELECT DISTINCT description, merchant, child_category, date FROM transactions WHERE\n      {{LLM('would this gift be appropriate for a child?', 'transactions::description')}} = 1\n      AND child_category = 'Gifts'",
                "question": "Show gifts for my kid",
                "serialized_db": "CREATE TABLE \"transactions\" (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n)"
            }
        ],
        "program": "\n{{#system~}}\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{~/system}}\n\n{{#user~}}\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL:\n{{~/user}}\n\n{{#assistant~}}\n{{gen \"result\" temperature=0.0}}\n{{~/assistant}}\n",
        "question": "how many players made an appearance that year?",
        "serialized_db": "CREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"player\" TEXT,\n  \"-1967\" TEXT,\n  \"03.30\" TEXT,\n  \"03.31\" TEXT,\n  \"04.04\" TEXT,\n  \"10.14\" TEXT,\n  \"1968\" TEXT,\n  \"total\" TEXT\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id             player  -1967 03.30 03.31 04.04 10.14 1968  total\n     0       0    shigeo yaegashi 41(11)     o     o     o     o 4(0) 45(11)\n     1       1      mitsuo kamata  37(2)     o  none     o     o 3(0)  40(2)\n     2       2 masakatsu miyamoto  36(1)  none     o  none     o 2(0)  38(1)\n*/"
    },
    "db_path": "./research/tmp/26258ef4-49b5-406a-84c0-50271e2aa092.db",
    "pred_text": [
        12
    ],
    "pred_sql": "SELECT COUNT(DISTINCT player) FROM w WHERE {{LLM('did this player make an appearance this year?', 'w::total')}} = 1"
}
```
5) Unnecessary use of BlendSQL
```json 
{
    "id": "nt-683",
    "question": "what is the total of deaths in internment camps and during the wartime flight?",
    "table_id": "csv/203-csv/234.csv",
    "table": {
        "page_title": "Demographic estimates of the flight and expulsion of Germans",
        "header": [
            "Description",
            "Total",
            "Poland",
            "Baltic States(Resettled in Poland during war)",
            "Resettled in Poland during war(from USSR and Romania)",
            "Sudetendeutsch",
            "SE Europe\\n(Hungary, Romania, Yugoslavia & Slovakia)"
        ],
        "rows": [
            [
                "Total Population Before Flight & Expulsion",
                "16,199,086",
                "11,038,826",
                "145,615",
                "365,622",
                "3,160,216",
                "1,488,807"
            ],
            [
                "Violent Deaths",
                "58,256",
                "44,603",
                "383",
                "747",
                "5,596",
                "6,927"
            ],
            [
                "Suicides",
                "14,356",
                "10,330",
                "157",
                "84",
                "3,411",
                "374"
            ],
            [
                "Deported( Forced Labor USSR)",
                "49,542",
                "32,947",
                "1,566",
                "6,465",
                "705",
                "7,859"
            ],
            [
                "In Internment Camps",
                "80,522",
                "27,847",
                "952",
                "1,037",
                "6,615",
                "44,071"
            ],
            [
                "During the wartime Flight",
                "93,283",
                "86,860",
                "2,394",
                "738",
                "629",
                "2,662"
            ],
            [
                "In the course of the Expulsions",
                "63,876",
                "57,814",
                "3,510",
                "561",
                "1,481",
                "510"
            ],
            [
                "Cause undetermined",
                "112,612",
                "106,991",
                "64",
                "3,116",
                "379",
                "2,062"
            ],
            [
                "Other Misc.",
                "566",
                "-",
                "38",
                "141",
                "73",
                "314"
            ],
            [
                "Total Confirmed Deaths",
                "473,013",
                "367,392",
                "9,064",
                "12,889",
                "18,889",
                "64,779"
            ],
            [
                "Total Unsolved Cases",
                "1,905,991",
                "1,404,993",
                "19,374",
                "73,794",
                "287,169",
                "120,661"
            ]
        ]
    },
    "sql": "SELECT SUM ( `Total` ) FROM w WHERE `Description` IN ( 'in internment camps' , 'during the wartime flight' )",
    "answer_text": [
        "173,805"
    ],
    "input_program_args": {
        "bridge_hints": "description ( during the wartime flight , in internment camps )",
        "examples": [
            {
                "blendsql": "SELECT year FROM w WHERE opponent = 'elizabeth matory'",
                "question": "What year was Elizabeth Matory the opponent of Charles Albert Ruppersberger?",
                "serialized_db": "\nTable Description: Dutch Ruppersberger (Electoral history)\nCREATE TABLE w(\n\trow_id int,\n\tyear int,\n\toffice text,\n\telection text,\n\tfilledcolumnname real,\n\tsubject text,\n\tparty text,\n\tvotes text,\n\t% text,\n\tfilledcolumnname_2 real,\n\topponent text,\n\tparty_2 text,\n\tvotes_2 text,\n\t%_2 text)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3;\nrow_id\tyear\toffice\telection\tfilledcolumnname\tsubject\tparty\tvotes\t%\tfilledcolumnname_2\topponent\tparty_2\tvotes_2\t%_2\n0\t1994\tbaltimore county executive\tgeneral\tnan\tdutch ruppersberger\tdemocratic\tn/a\tn/a\tnan\tn/a\tn/a\tn/a\tn/a\n1\t1998\tbaltimore county executive\tgeneral\tnan\tdutch ruppersberger\tdemocratic\t166482\t70.47\tnan\tjohn j. bishop\trepublican\t69449\t29.4\n2\t2002\tnone\tgeneral\tnan\tdutch ruppersberger\tdemocratic\t105718\t54.16\tnan\thelen delich bentley\trepublican\t88954\t45.57\n*/\n"
            },
            {
                "blendsql": "SELECT system FROM w WHERE system IN ('game boy advance', 'msx (japan only)') ORDER BY japan LIMIT 1",
                "question": "Which system has a lower number for Japan of the virtual console systems: Game Boy Advance or the Japan-only console MSX?",
                "serialized_db": "\nTable Description: Virtual Console (Titles)\nCREATE TABLE w (\n\trow_id int,\n\tsystem text,\n\tjapan int,\n\t[[list of virtual console games for wii u (north america)|north  america]] real,\n\tpal region - europe real,\n\tpal region - australia real\n)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3;\nrow_id\tsystem\tjapan\t[[list of virtual console games for wii u (north america)|north  america]]\tpal region - europe\tpal region - australia\n0\tnes/famicom\t148\t94.0\t89.0\t89.0\n1\tsuper nes/super famicom\t101\t51.0\t49.0\t49.0\n2\tnintendo 64\t22\t21.0\t21.0\t21.0\n*/  \n            "
            },
            {
                "blendsql": "SELECT player FROM w WHERE \n            {{\n                LLM('Has crossed swords on its logo?'; 'w::gold coast titans')\n            }} = 1\n        ",
                "question": "What player was transferred from the team that has crossed swords on its logo to the Warrington Wolves in the 2018 season?",
                "serialized_db": "\nTable Description: 2018 Warrington Wolves season (Transfers | In)\nCREATE TABLE w(\n\trow_id int,\n\tplayer text,\n\tsigned from text,\n\tcontract length text,\n\tannounced text)\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3;\nrow_id\tplayer\tsigned from\tcontract length\tannounced\n0\tsitaleki akauola\tpenrith panthers\tp2y\t2017-08-01 00:00:00\n1\tbryson goodwin\tsouth sydney rabbitohs\tp2y\t2017-10-01 00:00:00\n2\ttyrone roberts\tgold coast titans\tp3y\t2017-10-01 00:00:00\n*/\nCREATE TABLE Images(\n\trow_id int,\n\tgold coast titans text)\n/*\nAll rows of the table:\nSELECT * FROM w;\nrow_id\tgold coast titans\n0\ta logo for the golden knights is painted on the beach.\n*/\n"
            },
            {
                "blendsql": "SELECT DISTINCT merchant FROM w WHERE\n    {{LLM('is this a pizza shop?', 'w::merchant'}} = 1\n    AND {{\n       LLM(\n           'what state is this transaction from? Choose -1 when N.A.',\n           'w::description',\n           example_outputs='TX;CA;MA;-1;'\n       )\n    }} = 'MA'\n    AND parent_category = 'Food'",
                "question": "Pizza shop merchants in MA",
                "serialized_db": "\nTable Description: transactions\nCREATE TABLE w (\n    \"index\" INTEGER,\n    \"description\" TEXT,\n    \"amount\" REAL,\n    \"merchant\" TEXT,\n    \"cash_flow\" TEXT,\n    \"parent_category\" TEXT,\n    \"child_category\" TEXT,\n    \"date\" TEXT,\n/*\n3 example rows:\nSELECT * FROM transactions LIMIT 3\n index                                    description  amount merchant cash_flow parent_category         child_category       date  confidence_score\n     0  Zelle payment from JUSTIN ORENIC BACucw5n1t14  2000.0    Zelle   Expense        Cash/ATM Online Payment Service 2022-06-20          0.698891\n     1 Zelle Transfer Conf  UJ7QJF7TE  Emilia Galindo  1500.0    Zelle   Expense        Cash/ATM Online Payment Service 2022-01-16          0.701658\n     2   Zelle payment from SOPHIA YANEZ WFCT0Q9R3CNR  1474.0    Zelle   Expense        Cash/ATM Online Payment Service 2022-03-27          0.861237\n*/\n)"
            }
        ],
        "program": "\nGenerate BlendSQL given the question, table, passages, image captions to answer the question correctly.\nBlendSQL is a superset of SQLite, which adds external function calls for information not found within native SQLite.\nThese external functions should be wrapped in double curly brackets.\n\nIf question-relevant column(s) contents are not suitable for SQL comparisons or calculations, map it to a new column with clean content by a new grammar:\n    `LLM('question', '{table}::{column})'`\n\nExamples:\n{{~#each examples}}\n{{this.serialized_db}}\nQuestion: {{this.question}}\nBlendSQL: {{this.blendsql}}\n{{/each}}\n\n{{serialized_db}}\n\n{{#if bridge_hints}}\nHere are some values that may be helpful:\n{{bridge_hints}}\n{{/if}}\n\nQuestion: {{question}}\nBlendSQL: {{gen \"result\" temperature=0.0}}\n",
        "question": "what is the total of deaths in internment camps and during the wartime flight?",
        "serialized_db": "Table Description: Demographic estimates of the flight and expulsion of Germans\nCREATE TABLE \"w\" (\n\"index\" INTEGER,\n  \"row_id\" INTEGER,\n  \"description\" TEXT,\n  \"total\" INTEGER,\n  \"poland\" TEXT,\n  \"baltic states(resettled in poland during war)\" INTEGER,\n  \"resettled in poland during war(from ussr and romania)\" INTEGER,\n  \"sudetendeutsch\" INTEGER,\n  \"se europe\\n(hungary, romania, yugoslavia & slovakia)\" INTEGER\n)\n\n/*\n3 example rows:\nSELECT * FROM w LIMIT 3\n index  row_id                                description    total   poland  baltic states(resettled in poland during war)  resettled in poland during war(from ussr and romania)  sudetendeutsch  se europe\\n(hungary, romania, yugoslavia & slovakia)\n     0       0 total population before flight & expulsion 16199086 11038826                                         145615                                                 365622         3160216                                               1488807\n     1       1                             violent deaths    58256    44603                                            383                                                    747            5596                                                  6927\n     2       2                                   suicides    14356    10330                                            157                                                     84            3411                                                   374\n*/"
    },
    "db_path": "./research/tmp/431a7265-d77f-4fdc-b506-3293988744ce.db",
    "pred_text": [
        173805
    ],
    "pred_sql": " SELECT SUM(\"total\") FROM w WHERE \n            {{\n                LLM('is this a death during the wartime flight or in internment camps?', 'w::description', example_outputs='during the wartime flight;in internment camps;')\n            }} IN ('during the wartime flight', 'in internment camps')",
    "pred_has_ingredient": true
}
```