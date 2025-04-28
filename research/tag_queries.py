TAG_DATASET = [
    {
        "Query ID": 0,
        "DB used": "california_schools",
        "Query": "Among the schools with the average score in Math over 560 in the SAT test, how many schools are in counties in the bay area?",
        "(TAG baseline) Text2SQL Input": "List the schools with the average score in Math over 560 in the SAT test.",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "71",
        "BlendSQL": """SELECT COUNT(DISTINCT s.CDSCode) 
                FROM schools s 
                JOIN satscores sa ON s.CDSCode = sa.cds 
                WHERE sa.AvgScrMath > 560 
                AND {{LLMMap('Is this a county in the California Bay Area?', 's::County')}} = TRUE""",
        "Notes": None,
    },
    {
        "Query ID": 1,
        "DB used": "california_schools",
        "Query": "What is the telephone number for the school with the lowest average score in reading in a county in Southern California?",
        "(TAG baseline) Text2SQL Input": "List the telephone number for school ordered by average score in reading ascending",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "(562) 944-0033",
        "BlendSQL": """SELECT s.Phone 
            FROM satscores ss 
            JOIN schools s ON ss.cds = s.CDSCode 
            WHERE {{LLMMap('Is this county in Southern California?', 's::County')}} = TRUE
            AND ss.AvgScrRead IS NOT NULL
            ORDER BY ss.AvgScrRead ASC 
            LIMIT 1""",
        "Notes": None,
    },
    {
        "Query ID": 3,
        "DB used": "california_schools",
        "Query": "How many test takers are there at the school/s in a county with population over 2 million?",
        "(TAG baseline) Text2SQL Input": "How many test takers are there at the school/s in a county with population over 2 million?",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "244742",
        "BlendSQL": """SELECT SUM("NumTstTakr") AS TotalTestTakers
        FROM satscores
        JOIN schools ON schools.CDSCode = satscores.cds
        WHERE {{
            LLMMap(
                'What is the population of this California county? Give your best guess.',
                'schools::County'
            )
        }} > 2000000""",
        "Notes": None,
    },
    {
        "Query ID": 4,
        "DB used": "california_schools",
        "Query": "What is the grade span offered in the school with the highest longitude in cities in that are part of the 'Silicon Valley' region?",
        "(TAG baseline) Text2SQL Input": "What is the grade span offered in schools ordered by highest longitude",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "K-5",
        "BlendSQL": """SELECT GSoffered
            FROM schools 
            WHERE {{LLMMap('Is this county in Silicon Valley?', 'schools::County')}} = TRUE
            ORDER BY "Longitude" DESC 
            LIMIT 1""",
        "Notes": None,
    },
    {
        "Query ID": 5,
        "DB used": "california_schools",
        "Query": "What are the two most common first names among the female school administrators?",
        "(TAG baseline) Text2SQL Input": "List the names of school administrators from most common to least common",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": ["Jennifer", "Lisa"],
        "order_insensitive_answer": True,
        "BlendSQL": """WITH top_names AS (
            SELECT AdmFName1 AS name, COUNT(AdmFName1) AS count
            FROM schools 
            GROUP BY "AdmFName1" 
            ORDER BY count DESC 
            LIMIT 20
        ) SELECT name FROM top_names 
        WHERE {{LLMMap('Is this a female name?', 'top_names::name')}} = TRUE
        ORDER BY count DESC LIMIT 2""",
        "Notes": "Works, assuming that two of the top 20 names are female names. Otherwise would need to apply LLM function over entire table - but, that's the 'correct' interpretation of the query. TAG bench uses only the top 20 names, like we do here.",
    },
    {
        "Query ID": 6,
        "DB used": "codebase_community",
        "Query": "Among the posts owned by csgillespie, how many of them are root posts and mention academic papers?",
        "(TAG baseline) Text2SQL Input": "List the body of the root posts owned by csgillespie",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "4",
        "BlendSQL": """SELECT COUNT(*)
        FROM posts 
        WHERE OwnerUserId = (SELECT Id FROM users WHERE DisplayName = 'csgillespie') 
        AND ParentId IS NULL 
        AND {{LLMMap('Does this post mention academic papers?', 'posts::Body')}} = TRUE""",
        "Notes": None,
    },
    {
        "Query ID": 8,
        "DB used": "codebase_community",
        "Query": "How many of the comments with a score of 17 are about statistics?",
        "(TAG baseline) Text2SQL Input": "List the comments with a score of 17.",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "4",
        "BlendSQL": """SELECT COUNT(*) 
        FROM comments 
        WHERE Score = 17 
        AND {{LLMMap('Is this text about statistics?', 'comments::Text')}} = TRUE""",
        "Notes": None,
    },
    {
        "Query ID": 10,
        "DB used": "codebase_community",
        "Query": "Of the posts with views above 80000, how many discuss the R programming language?",
        "(TAG baseline) Text2SQL Input": "List the bodies of the posts with views above 80000",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "3",
        "BlendSQL": """SELECT COUNT(*) 
        FROM posts 
        WHERE ViewCount > 80000 
        AND {{LLMMap('Does this text discuss the R programming language?', 'posts::Body')}} = TRUE""",
        "Notes": None,
    },
    {
        "Query ID": 11,
        "DB used": "formula_1",
        "Query": "Please give the names of the races held on the circuits in the middle east.",
        "(TAG baseline) Text2SQL Input": "List the names of races on circuits and their locations",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": [
            "Bahrain Grand Prix",
            "Turkish Grand Prix",
            "Abu Dhabi Grand Prix",
            "Azerbaijan Grand Prix",
            "European Grand Prix",
        ],
        "order_insensitive_answer": True,
        "BlendSQL": """SELECT DISTINCT r.name 
        FROM races r 
        JOIN circuits c ON r.circuitId = c.circuitId 
        WHERE {{LLMMap('Is this a country in the Middle East?', 'c::country')}} = TRUE""",
        "Notes": "Is Europe in the Middle East?",
    },
    {
        "Query ID": 13,
        "DB used": "formula_1",
        "Query": "How many Asian drivers competed in the 2008 Australian Grand Prix?",
        "(TAG baseline) Text2SQL Input": "List the race of the drivers who competed in the 2008 Australian Grand Prix",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "2",
        "BlendSQL": """SELECT COUNT(DISTINCT d.driverId) AS asian_driver_count
        FROM drivers d
        JOIN results r ON d.driverId = r.driverId
        JOIN races ra ON r.raceId = ra.raceId
        WHERE ra.year = 2008 AND ra.name = 'Australian Grand Prix' AND {{LLMMap('Is this nationality Asian?', 'd::nationality')}} = TRUE""",
        "Notes": None,
    },
    {
        "Query ID": 16,
        "DB used": "european_football_2",
        "Query": "What is the preferred foot when attacking of the player with the most Ballon d'Or awards of all time?",
        "(TAG baseline) Text2SQL Input": "What is the preferred foot when attacking of the player with the most Ballon d'Or awards of all time?",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "left",
        "BlendSQL": """SELECT preferred_foot FROM Player JOIN Player_Attributes ON Player.player_api_id = Player_Attributes.player_api_id
            WHERE player_name = {{
                LLMQA(
                    "Which player has the most Ballon d'Or awards?"
                )
            }} LIMIT 1""",
        "Notes": None,
    },
    {
        "Query ID": 18,
        "DB used": "european_football_2",
        "Query": "List the football player with a birthyear of 1970 who is an Aquarius",
        "(TAG baseline) Text2SQL Input": "List the football player with a birthyear of 1970 who is an Aquarius",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "Hans Vonk",
        "BlendSQL": """WITH DateRange AS (
SELECT * FROM VALUES {{LLMQA('What are the start and end date ranges for an Aquarius? Respond in MM-DD.', regex='\\d{2}-\\d{2}', modifier='{2}')}}
)
SELECT player_name FROM Player
WHERE birthday LIKE '1970%'
AND strftime('%m-%d', birthday) >= (SELECT min(column1, column2) FROM DateRange)
AND strftime('%m-%d', birthday) <= (SELECT max(column1, column2) FROM DateRange)""",
        "Notes": None,
    },
    {
        "Query ID": 19,
        "DB used": "european_football_2",
        "Query": "Please list the league from the country which is landlocked.",
        "(TAG baseline) Text2SQL Input": "Please list the unique leagues and the country they are from",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "Switzerland Super League",
        "BlendSQL": """SELECT l.name FROM League l 
        JOIN Country c ON l.country_id = c.id
        WHERE {{LLMMap('Is this country landlocked?', 'c::name')}} = TRUE""",
        "Notes": "Broken as of d2af6ed, due to JOIN on duplicate column names",
    },
    {
        "Query ID": 20,
        "DB used": "european_football_2",
        "Query": "How many matches in the 2008/2009 season were held in countries where French is an official language?",
        "(TAG baseline) Text2SQL Input": "List the matches from the 2008/2009 season and the countries they were held in",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "866",
        "BlendSQL": """SELECT COUNT(*) FROM "Match" m
        JOIN Country c ON m.country_id = c.id
        WHERE m.season = '2008/2009' 
        AND c.name IN {{LLMQA('In which of these countries is French an official language?', options='c::name')}}
        """,
        "Notes": None,
    },
    {
        "Query ID": 21,
        "DB used": "european_football_2",
        "Query": "Of the top three away teams that scored the most goals, which one has the most fans?",
        "(TAG baseline) Text2SQL Input": "List the top three away teams that scored the most goals",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "FC Barcelona",
        "BlendSQL": """WITH top_teams AS (
SELECT DISTINCT team_long_name AS name, away_team_Goal FROM Team t 
JOIN "Match" m ON t.team_api_id = m.away_team_api_id
ORDER BY away_team_goal DESC LIMIT 3
) SELECT {{LLMQA('Which team has the most fans?', options='top_teams::name')}}""",
        "Notes": None,
    },
    {
        "Query ID": 24,
        "DB used": "debit_card_specializing",
        "Query": "Which year recorded the most gas use paid in the higher value currency?",
        "(TAG baseline) Text2SQL Input": "Which year recorded the most gas use paid in the higher value currency?",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "2013",
        "BlendSQL": """SELECT ym.Date / 100 AS "year" FROM customers c
        JOIN yearmonth ym ON c.CustomerID = ym.CustomerID
        WHERE c.Currency = {{LLMQA('Which currency is the higher value?')}}
        GROUP BY "year"
        ORDER BY SUM(ym.Consumption) DESC LIMIT 1
        """,
        "Notes": None,
    },
    {
        "Query ID": 108,
        "DB used": "codebase_community",
        "Query": "Among the posts that were voted by user 1465, determine if the post is relevant to machine learning. Respond with YES if it is and NO if it is not.",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": ["YES", "YES", "YES"],
        "order_insensitive_answer": True,
        "BlendSQL": """SELECT {{LLMMap('Is the post relevant to Machine Learning?', 'posts::Body', options='YES;NO')}} 
        FROM posts JOIN votes v ON posts.Id = v.PostId WHERE v.UserId = 1465
        """,
        "Notes": None,
    },
    {
        "Query ID": 109,
        "DB used": "codebase_community",
        "Query": "Extract the statistical term from the post titles which were edited by Vebjorn Ljosa.",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": [
            "beta-binomial distribution",
            "AdaBoost",
            "SVM",
            "Kolmogorov-Smirnov statistic",
        ],
        "order_insensitive_answer": True,
        "BlendSQL": """SELECT {{LLMMap('Extract the most statistical term from the title', 'p::Title', output_type='substring')}}
        FROM posts p JOIN users u ON p.OwnerUserId = u.Id WHERE u.DisplayName = 'Vebjorn Ljosa'""",
        "Notes": None,
    },
    {
        "Query ID": 110,
        "DB used": "codebase_community",
        "Query": "List the Comment Ids of the positive comments made by the top 5 newest users on the post with the title 'Analysing wind data with R'",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": ["11449"],
        "order_insensitive_answer": True,
        "BlendSQL": """WITH new_user_comments AS
        (
            SELECT c.Id, c.Text FROM comments c
            JOIN posts p ON p.Id = c.PostId
            JOIN users u ON p.OwnerUserId = u.Id
            WHERE p.Title = 'Analysing wind data with R'
            ORDER BY u.CreationDate LIMIT 5
        ) SELECT Id FROM new_user_comments new_c WHERE {{LLMMap('Does the comment have a positive sentiment?', 'new_c::Text')}} = TRUE""",
        "Notes": None,
    },
    {
        "Query ID": 111,
        "DB used": "codebase_community",
        "Query": 'For the post from which the tag "bayesian" is excerpted from, identify whether the body of the post is True or False. Answer with True or False ONLY.',
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "True",
        "BlendSQL": """{{
            LLMQA(
                'Is the content in "Body" true or false?',
                (
                    SELECT Body FROM posts p 
                    JOIN tags t ON t.ExcerptPostId = p.Id 
                    WHERE t.TagName = 'bayesian'
                ), 
                options='True;False'
            )
        }}
        """,
        "Notes": None,
    },
    {
        "Query ID": 25,
        "DB used": "debit_card_specializing",  # Fixed from original TAG (was `codebase_community`)
        "Query": "What is the average total price of the transactions taken place in gas stations in the country which is historically known as Bohemia, to the nearest integer?",
        "(TAG baseline) Text2SQL Input": "What is the average total price of the transactions taken place in gas stations in the country which is historically known as Bohemia, to the nearest integer?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "453",
        "BlendSQL": """SELECT CAST(ROUND(AVG(t.Price)) AS INT) FROM transactions_1k t 
        JOIN gasstations g ON g.GasStationID = t.GasStationID
        WHERE g.Country = {{LLMQA('Which is the abbreviation for the country historically known as Bohemia')}} 
        """,
        "Notes": "Good example of program-inferred constraints.",
    },
    {
        "Query ID": 27,
        "DB used": "codebase_community",
        "Query": "List the username of the oldest user located in the capital city of Austria who obtained the Supporter badge?",
        "(TAG baseline) Text2SQL Input": "List the username of the oldest user located in the capital city of Austria who obtained the Supporter badge?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "ymihere",
        "BlendSQL": """SELECT DisplayName FROM users u
        JOIN badges b ON u.Id = b.UserId
        WHERE b.Name = 'Supporter'
        AND u.Location = {{LLMQA("What's the capital city of Austria?")}}
        ORDER BY u.Age DESC LIMIT 1
        """,
        "Notes": "TAG code is incorrect. There are many different ways 'Vienna' is represented in the database: ['Vienna, Austria', 'Vienna/Austria', 'vienna', 'Vienna, VA']. Though it is impossible to determine which are pointing to the location in Austria.",
    },
    {
        "Query ID": 29,
        "DB used": "debit_card_specializing",
        "Query": "What is the difference in gas consumption between customers who pay using the currency of the Czech Republic and who pay the currency of European Union in 2012, to the nearest integer?",
        "(TAG baseline) Text2SQL Input": "What is the difference in gas consumption between customers who pay using the currency of the Czech Republic and who pay the currency of European Union in 2012, to the nearest integer?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "402524570",
        "BlendSQL": """WITH gas_consumption AS (
            SELECT ym.Consumption, c.Currency FROM customers c 
            JOIN yearmonth ym ON c.CustomerID = ym.CustomerID
            WHERE ym.Date / 100 = 2012
        ) SELECT 
        CAST(ROUND((SELECT SUM(Consumption) FROM gas_consumption g WHERE g.Currency = {{LLMQA('Currency code of Czech Republic?')}}) - 
        (SELECT SUM(Consumption) FROM gas_consumption g WHERE g.Currency = {{LLMQA('Currency code of European Union?')}})) AS INT)
        """,
        "Notes": None,
    },
    {
        "Query ID": 30,
        "DB used": "debit_card_specializing",
        "Query": "Is it true that more SMEs pay in Czech koruna than in the second-largest reserved currency in the world?",
        "(TAG baseline) Text2SQL Input": "Is it true that more SMEs pay in Czech koruna than in the second-largest reserved currency in the world?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "Yes",
        "BlendSQL": """WITH sme_payers AS (
        SELECT * FROM customers c WHERE c.Segment = 'SME'
        ) SELECT CASE 
        WHEN (SELECT COUNT(*) FROM sme_payers p WHERE p.Currency = 'CZK') >
        (SELECT COUNT(*) FROM sme_payers p WHERE p.Currency = {{LLMQA('What is the 3 letter code for the second-largest reserved currency in the')}})
        THEN 'Yes' ELSE 'No' END
        """,
        "Notes": None,
    },
    {
        "Query ID": 33,
        "DB used": "california_schools",
        "Query": "What is the total number of schools whose total SAT scores are greater or equal to 1500 whose mailing city is the county seat of Lake County, California?",
        "(TAG baseline) Text2SQL Input": "What is the total number of schools whose total SAT scores are greater or equal to 1500 whose mailing city is the county seat of Lake County, California?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "2",
        "BlendSQL": """
        SELECT COUNT(*) FROM schools s 
        JOIN satscores ss ON ss.cds = s.CDSCode
        WHERE s.City = {{LLMQA('What is the name of the city that is the county seat of Lake County, California?')}}
        AND ss.AvgScrMath + ss.AvgScrWrite + ss.AvgScrRead >= 1500
        """,
        "Notes": None,
    },
    {
        "Query ID": 35,
        "DB used": "formula_1",
        "Query": "How many drivers born after the year of Vietnam War have been ranked 2?",
        "(TAG baseline) Text2SQL Input": "How many drivers born after the year of Vietnam War have been ranked 2?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "27",
        "BlendSQL": """SELECT COUNT(DISTINCT d.driverId) FROM drivers d 
        JOIN results r ON d.driverId = r.driverId 
        WHERE r.rank = 2 
        AND CAST(SUBSTR(d.dob, 1, 4) AS NUMERIC) > {{LLMQA('What year did the Vietnam war end?', regex='\d{4}')}}
        """,
        "Notes": None,
    },
    {
        "Query ID": 36,
        "DB used": "formula_1",
        "Query": "Among all European Grand Prix races, what is the percentage of the races were hosted in the country where the Bundesliga happens, to the nearest whole number?",
        "(TAG baseline) Text2SQL Input": "Among all European Grand Prix races, what is the percentage of the races were hosted in the country where the Bundesliga happens, to the nearest whole number?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "52",
        "BlendSQL": """WITH gp_races AS (
        SELECT country FROM races r 
        JOIN circuits c ON c.circuitId = r.circuitId 
        WHERE r.name = 'European Grand Prix'
        ) SELECT CAST(ROUND(1.0 *
        (SELECT COUNT(*) FROM gp_races WHERE gp_races.country = {{LLMQA('Where does the Bundesliga happen?')}}) / 
        (SELECT COUNT(*) FROM gp_races) * 100) AS INT)
        """,
        "Notes": None,
    },
    {
        "Query ID": 37,
        "DB used": "european_football_2",
        "Query": "From 2010 to 2015, what was the average overall rating, rounded to the nearest integer, of players who are higher than 170 and shorter than Michael Jordan?",
        "(TAG baseline) Text2SQL Input": "From 2010 to 2015, what was the average overall rating, rounded to the nearest integer, of players who are higher than 170 and shorter than Michael Jordan?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "69",
        "BlendSQL": """SELECT CAST(ROUND(AVG(pa.overall_rating)) AS INT)
        FROM Player_Attributes pa 
        JOIN Player p ON p.player_api_id = pa.player_api_id 
        WHERE p.height > 170 
        AND p.height < {{LLMQA('How tall was Michael Jordan in cm? Give your best guess.')}}
        AND CAST(SUBSTR(pa.date, 1, 4) AS NUMERIC) BETWEEN 2010 AND 2015
        """,
        "Notes": None,
    },
    {
        "Query ID": 38,
        "DB used": "formula_1",
        "Query": "Among the drivers that finished the race in the 2008 Australian Grand Prix, how many debuted earlier than Lewis Hamilton?",
        "(TAG baseline) Text2SQL Input": "List the names of the drivers who finished the race in the 2008 Australian Grand Prix",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "3",
        "BlendSQL": """WITH "2008_gp_drivers" AS (
        SELECT CONCAT(d.forename, ' ', d.surname) AS name FROM drivers d
        JOIN results r ON r.driverId = d.driverId 
        JOIN races ra ON r.raceId = ra.raceId
        WHERE ra.name = 'Australian Grand Prix'
        AND ra.year = 2008 
        ) SELECT COUNT(*) FROM "2008_gp_drivers"
        WHERE {{LLMMap('What year did this driver debut?', '2008_gp_drivers::name', output_type='int')}} > {{LLMQA('What year did Lewis Hamilton debut in F1?', output_type='int')}}
        """,
        "Notes": "TAG seems wrong? No mention of Lewis Hamilton.",
    },
    {
        "Query ID": 39,
        "DB used": "european_football_2",
        "Query": "How many players were born after the year of the 14th FIFA World Cup?",
        "(TAG baseline) Text2SQL Input": "How many players were born after the year of the 14th FIFA World Cup?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "3028",
        "BlendSQL": """SELECT COUNT(*) FROM Player p 
        WHERE CAST(SUBSTR(birthday, 1, 4) AS NUMERIC) > {{LLMQA('What year did the 14th FIFA World Cup take place?', regex='\d{4}')}}
        """,
        "Notes": "Gets wrong year for 14th FIFA World Cup.",
    },
    {
        "Query ID": 40,
        "DB used": "european_football_2",
        "Query": "Among the players whose height is over 180, how many of them have a volley score of over 70 and are taller than Bill Clinton?",
        "(TAG baseline) Text2SQL Input": "Among the players whose height is over 180, how many of them have a volley score of over 70 and are taller than Bill Clinton?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "88",
        "BlendSQL": """SELECT COUNT(DISTINCT p.player_api_id) FROM Player p 
        JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
        WHERE p.height >= 180 
        AND pa.volleys > 70
        AND p.height > {{LLMQA('How tall is Bill Clinton in centimeters?')}}
        """,
        "Notes": "Gets Bill Clinton height wrong",
    },
    {
        "Query ID": 41,
        "DB used": "california_schools",
        "Query": "Give the number of schools with the percent eligible for free meals in K-12 is more than 0.1 and test takers whose test score is greater than or equal to the score one hundred points less than the maximum.",
        "(TAG baseline) Text2SQL Input": "Give the number of schools with the percent eligible for free meals in K-12 is more than 0.1 and test takers whose test score is greater than or equal to the score one hundred points less than the maximum.",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "1",
        "BlendSQL": """SELECT COUNT(DISTINCT f.CDSCode) FROM frpm f 
        JOIN satscores ss ON ss.cds = f.CDSCode
        WHERE f."Free Meal Count (K-12)" / f."Enrollment (K-12)" > 0.1
        AND ss.AvgScrRead + ss.AvgScrMath >= {{LLMQA('What is the maximum possible SAT score?')}} - 300
        """,
        "Notes": "TAG query seems wrong, they subtract 300 instead of 100.",
    },
    {
        "Query ID": 42,
        "DB used": "california_schools",
        "Query": "How many schools have the difference in enrollements between K-12 and ages 5-17 as more than the number of days in April?",
        "(TAG baseline) Text2SQL Input": "How many schools have the difference in enrollements between K-12 and ages 5-17 as more than the number of days in April?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "1236",
        "BlendSQL": """SELECT COUNT(DISTINCT s.CDSCode) FROM frpm f 
        JOIN schools s ON s.CDSCode = f.CDSCode
        WHERE (f."Enrollment (K-12)" - f."Enrollment (Ages 5-17)") > {{LLMQA('How many days are in April?')}}
        """,
        "Notes": "Ground truth answer seems wrong - looks like it should be 1239?",
    },
    {
        "Query ID": 43,
        "DB used": "codebase_community",
        "Query": "Among the users who have more than 100 upvotes, how many of them are older than the median age in America?",
        "(TAG baseline) Text2SQL Input": "Among the users who have more than 100 upvotes, how many of them are older than the median age in America?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "32",
        "BlendSQL": """SELECT COUNT(DISTINCT u.Id) FROM users u 
        WHERE u.UpVotes > 100
        AND u.Age > {{LLMQA('What is the median age in America? Give your best guess.')}}
        """,
        "Notes": None,
    },
    {
        "Query ID": 44,
        "DB used": "european_football_2",
        "Query": "Please list the player names taller than 6 foot 8?",
        "(TAG baseline) Text2SQL Input": "Please list the player names taller than 6 foot 8?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": ["Kristof van Hout"],
        "order_insensitive_answer": True,
        "BlendSQL": """SELECT DISTINCT p.player_name FROM Player p
        WHERE p.height > {{LLMQA('What is 6 foot 8 in centimeters?')}}
        """,
        "Notes": None,
    },
    {
        "Query ID": 45,
        "DB used": "european_football_2",
        "Query": "How many players whose first names are Adam and weigh more than 77.1kg?",
        "(TAG baseline) Text2SQL Input": "How many players whose first names are Adam and weigh more than 77.1kg?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "24",
        "BlendSQL": """SELECT COUNT(*) FROM Player p 
        WHERE p.player_name LIKE 'Adam%'
        AND p.weight > {{LLMQA('What is 77.1kg in pounds?')}}
        """,
        "Notes": None,
    },
    {
        "Query ID": 46,
        "DB used": "european_football_2",
        "Query": "Please provide the names of top three football players who are over 5 foot 11 tall in alphabetical order.",
        "(TAG baseline) Text2SQL Input": "Please provide the names of top three football players who are over 5 foot 11 tall in alphabetical order.",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": ["Aaron Appindangoye", "Aaron Galindo", "Aaron Hughes"],
        "order_insensitive_answer": False,
        "BlendSQL": """SELECT player_name FROM Player p 
        WHERE p.height > {{LLMQA('What is 5 foot 11 in centimeters?')}}
        ORDER BY player_name LIMIT 3
        """,
        "Notes": None,
    },
    {
        "Query ID": 47,
        "DB used": "debit_card_specializing",
        "Query": "How many transactions taken place in the gas station in the Czech Republic are with a price of over 42.74 US dollars?",
        "(TAG baseline) Text2SQL Input": "How many transactions taken place in the gas station in the Czech Republic are with a price of over 42.74 US dollars?",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "56",
        "BlendSQL": """SELECT COUNT(*) FROM transactions_1k t 
        JOIN gasstations gs ON t.GasStationID = gs.GasStationID 
        WHERE gs.Country = 'CZE'
        AND t.Price > {{LLMQA('What is 45 USD in CZK?')}}
        """,
        "Notes": "Currency conversion is wrong",
    },
    {
        "Query ID": 48,
        "DB used": "formula_1",
        "Query": "Which of these circuits is located closer to a capital city, Silverstone Circuit, Hockenheimring or Hungaroring?",
        "(TAG baseline) Text2SQL Input": "List the names and latitude and longitude of the circuits Silverstone Circuit, Hockenheimring and Hungaroring",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "Hungaroring",
        "BlendSQL": """{{
            LLMQA(
                'Which is located closer to a capital city?', 
                options='Silverstone Circuit;Hockenheimring;Hungaroring'
            )
            
        }}
        """,
        "Notes": None,
    },
    {
        "Query ID": 49,
        "DB used": "formula_1",
        "Query": "Which race was Alex Yoong in when he was in the top half of finishers?",
        "(TAG baseline) Text2SQL Input": "List the races and positions for Alex Yoong",
        "Query type": "Comparison",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "Australian Grand Prix",
        "BlendSQL": """SELECT ra.name FROM drivers d
        JOIN results r ON d.driverId = r.driverId
        JOIN races ra ON ra.raceId = r.raceId 
        WHERE d.forename = 'Alex' AND d.surname = 'Yoong'
        AND r.position < {{LLMQA('How many starting positions are typically in an F1 race?')}} / 2
        """,
        "Notes": "LLMQA is wrong (22 instead of 20), but this database results in the correct answer, despite returning multiple values.",
    },
    {
        "Query ID": 50,
        "DB used": "california_schools",
        "Query": "Among the magnet schools with SAT test takers of over 500, which school name sounds most futuristic?",
        "(TAG baseline) Text2SQL Input": "List the names of the magnet schools with SAT test takers of over 500",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "Polytechnic High",
        "BlendSQL": """{{
            LLMQA(
                'Which school name sounds the most futuristic?',
                options=(
                    SELECT s.School FROM schools s 
                    JOIN satscores ss ON s.CDSCode = ss.cds
                    WHERE s.Magnet = TRUE
                    AND ss.NumTstTakr > 500
                )
            )
        }}""",
        "Notes": "'Most Futuristic' feels incredibly subjective here.",
    },
    {
        "Query ID": 51,
        "DB used": "codebase_community",
        "Query": "Of the 5 posts wih highest popularity, list their titles in order of most technical to least technical.",
        "(TAG baseline) Text2SQL Input": "List the body of the 5 posts with the highest popularity",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": [
            "How to interpret and report eta squared / partial eta squared in statistically significant and non-significant analyses?",
            "How to interpret F- and p-value in ANOVA?",
            "What is the meaning of p values and t values in statistical tests?",
            "How to choose between Pearson and Spearman correlation?",
            "How do I get the number of rows of a data.frame in R?",
        ],
        "BlendSQL": """WITH top_posts AS (
            SELECT Title FROM posts p 
            ORDER BY p.ViewCount DESC 
            LIMIT 5
        ) SELECT * FROM VALUES {{LLMQA('Order the article titles, from most technical to least technical', options='top_posts::Title')}} 
        """,
        "Notes": "Again, 'Most technical' is very subjective.",
    },
    {
        "Query ID": 52,
        "DB used": "codebase_community",
        "Query": "What are the Post Ids of the top 2 posts in order of most grateful comments received on 9-14-2014",
        "(TAG baseline) Text2SQL Input": "List the post ids and comments for the posts on 9-14-2014",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": ["115372", "115254"],
        "order_insensitive_answer": False,
        "BlendSQL": """SELECT p.Id FROM posts p 
        JOIN comments c ON p.Id = c.PostId 
        WHERE c.CreationDate LIKE '2014-09-14%'
        AND {{LLMMap("Is this a grateful comment, saying things like 'Thank you'?", 'c::Text')}} = TRUE
        GROUP BY p.Id
        ORDER BY COUNT(c.Id) DESC
        LIMIT 2
        """,
        "Notes": None,
    },
    {
        "Query ID": 53,
        "DB used": "codebase_community",
        "Query": "For the post owned by csgillespie with the highest popularity, what is the most sarcastic comment?",
        "(TAG baseline) Text2SQL Input": "List the text of the comments on the post owned by csgillespie with the highest popularity",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "That pirates / global warming chart is clearly cooked up by conspiracy theorists - anyone can see they have deliberately plotted even spacing for unequal time periods to avoid showing the recent sharp increase in temperature as pirates are almost entirely wiped out.\nWe all know that as temperatures rise it makes the rum evaporate and pirates cannot survive those conditions.\n;-)",
        "BlendSQL": """WITH top_post AS (
        SELECT p.Id FROM posts p
        JOIN users u ON p.OwnerUserId = u.Id 
        WHERE u.DisplayName = 'csgillespie'
        ORDER BY p.ViewCount DESC LIMIT 1
        ) SELECT {{
            LLMQA(
                'Which of these comments is the most sarcastic?',
                options=(
                    SELECT c.Text FROM comments c 
                    JOIN top_post ON top_post.Id = c.PostId
                )
            )
        }}
        """,
        "Notes": "Ground truth is wrong - it contains a value not present in the database. It misses the final '\n;-)' bit. I've corrected it here.",
    },
    {
        "Query ID": 54,
        "DB used": "codebase_community",
        "Query": "Among the top 10 most popular tags, which is the least related to statistics?",
        "(TAG baseline) Text2SQL Input": "What are the top 10 most popular tags?",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "self-study",
        "BlendSQL": """WITH popular_tags AS (
        SELECT TagName FROM tags t 
        ORDER BY t.Count DESC LIMIT 10
        ) SELECT {{
            LLMQA(
                'Which of these tags is LEAST related to statistics?',
                options='popular_tags::TagName'
            )
        }}""",
        "Notes": None,
    },
    {
        "Query ID": 55,
        "DB used": "codebase_community",
        "Query": "Of the top 10 most favorited posts, what is the Id of the most lighthearted post?",
        "(TAG baseline) Text2SQL Input": "List the Id and body of the top 10 most favorited posts",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "423",
        "BlendSQL": """WITH favorited_posts AS (
        SELECT Id, Body FROM posts p 
        ORDER BY p.FavoriteCount DESC LIMIT 10
        ) SELECT Id FROM favorited_posts 
        WHERE Body = {{LLMQA('Which of these is the most lighthearted?')}}
        """,
        "Notes": """Questionable annotation. Ground truth post is:
        '<p>This is one of my favorites:</p>\n\n<p><img src="http://imgs.xkcd.com/comics/correlation.png" alt="alt text"></p>\n\n<p>One entry per answer. This is in the vein of the Stack Overflow question <em><a href="http://stackoverflow.com/questions/84556/whats-your-favorite-programmer-cartoon">What’s your favorite “programmer” cartoon?</a></em>.</p>\n\n<p>P.S. Do not hotlink the cartoon without the site\'s permission please.</p>\n'
        
        Predicted is:
        '<p>From <a href="http://en.wikipedia.org/wiki/Degrees_of_freedom_%28statistics%29#cite_note-1">Wikipedia</a>, there are three interpretations of the degrees of freedom of a statistic:</p>\n\n<blockquote>\n  <p>In statistics, the number of degrees of freedom is the number of\n  values in the <strong>final calculation</strong> of a statistic that are <strong>free to vary</strong>.</p>\n  \n  <p>Estimates of statistical parameters can be based upon different\n  amounts of information or data. The number of <strong>independent pieces of\n  information</strong> that go into the estimate of a parameter is called the\n  degrees of freedom (df). In general, the degrees of freedom of an\n  estimate of a parameter is equal to <strong>the number of independent scores\n  that go into the estimate</strong> minus <strong>the number of parameters used as\n  intermediate steps in the estimation of the parameter itself</strong> (which,\n  in sample variance, is one, since the sample mean is the only\n  intermediate step).</p>\n  \n  <p>Mathematically, degrees of freedom is <strong>the dimension of the domain of a\n  random vector</strong>, or essentially <strong>the number of \'free\' components: how\n  many components need to be known before the vector is fully\n  determined</strong>.</p>\n</blockquote>\n\n<p>The bold words are what I don\'t quite understand. If possible, some mathematical formulations will help clarify the concept.</p>\n\n<p>Also do the three interpretations agree with each other?</p>\n'
        
        Neither seem particularly lighthearted?

        """,
    },
    {
        "Query ID": 56,
        "DB used": "codebase_community",
        "Query": "Among the posts owned by a user over 65 with a score of over 10, what are the post id's of the top 2 posts made with the least expertise?",
        "(TAG baseline) Text2SQL Input": "List the post id and body of the posts owned by a user over 65 with a score of over 10",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": ["8485", "15670"],
        "order_insensitive_answer": True,
        "BlendSQL": """WITH filtered_posts AS (
            SELECT p.Id, p.Body FROM posts p
            JOIN users u ON p.OwnerUserId = u.Id
            WHERE u.Age > 65 AND p.Score > 10
        ) SELECT * FROM VALUES {{
            LLMQA(
                'Which 2 `Id` values are attached to the 2 posts whose authors have the least expertise?',
                context=(
                    SELECT * FROM filtered_posts
                ),
                options='filtered_posts::Id',
                quantifier='{2}'
            )
        }}
        """,
        "Notes": "What does 'written with the least expertise' mean?",
    },
    {
        "Query ID": 57,
        "DB used": "codebase_community",
        "Query": "Among the badges obtained by csgillespie in 2011, which is the most similar to an English grammar guide?",
        "(TAG baseline) Text2SQL Input": "List the names of the badges obtained by csgillespie in 2011.",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "Strunk & White",
        "BlendSQL": """WITH filtered_badges AS (
        SELECT b.Name FROM badges b 
        JOIN users u ON u.Id = b.UserId
        WHERE u.DisplayName = 'csgillespie'
        ) SELECT {{
            LLMQA(
                'Which is most similar to an English grammar guide?', 
                options='filtered_badges::Name'
            )
        }}
        """,
        "Notes": "How is 'Strunk & White' like an English grammar guide?",
    },
    {
        "Query ID": 58,
        "DB used": "codebase_community",
        "Query": "Of the posts owned by Yevgeny, what are the id's of the top 3 most pessimistic?",
        "(TAG baseline) Text2SQL Input": "List the post id and body of the posts owned by Yevgeny",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": ["23819", "24216", "35748"],
        "order_insensitive_answer": True,
        "BlendSQL": """WITH yevgeny_posts AS (
            SELECT p.Id, p.Body FROM posts p 
            JOIN users u ON p.OwnerUserId = u.Id
            WHERE u.DisplayName = 'Yevgeny'
        ) SELECT * FROM VALUES {{
            LLMQA(
                'Which 2 `Id` values are attached to the 3 most pessimistic comments?',
                context=(
                    SELECT * FROM yevgeny_posts
                ),
                options='yevgeny_posts::Id',
                quantifier='{3}'
            )
        }}""",
        "Notes": "Very long context passed to LLMQA here.",
    },
    {
        "Query ID": 59,
        "DB used": "european_football_2",
        "Query": "Of the top 10 players taller than 180 ordered by average heading accuracy descending, what are the top 3 most unique sounding names?",
        "(TAG baseline) Text2SQL Input": "List the names of the top 10 players taller than 180 ordered by average heading accuracy descending.",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": ["Naldo", "Per Mertesacker", "Didier Drogba"],
        "order_insensitive_answer": True,
        "BlendSQL": """WITH top_players AS (
            SELECT p.player_name, AVG(pa.heading_accuracy) AS avg_heading_accuracy FROM Player p 
            JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
            WHERE p.height > 180 
            GROUP BY p.player_api_id 
            ORDER BY avg_heading_accuracy DESC 
            LIMIT 10
        ) SELECT * FROM VALUES {{
        LLMQA(
            "Which 3 of these names could be said to be the 'most unique'?", 
            options="top_players::player_name", 
            quantifier="{3}"
            )
        }}
        """,
        "Notes": "'Unique sounding name' doesn't mean much. Why is 'Per Mertesacker' more unique than 'Miroslav Klose', etc.?",
    },
    {
        "Query ID": 60,
        "DB used": "codebase_community",
        "Query": "Out of users that have obtained at least 200 badges, what are the top 2 display names that seem most based off a real name?",
        "(TAG baseline) Text2SQL Input": "List the users who have obtained at least 200 badges",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": ["Jeromy Anglim", "Glen_b"],
        "order_insensitive_answer": True,
        "BlendSQL": """SELECT * FROM VALUES {{
            LLMQA(
                'Which 2 of these display names most based off of a real name?',
                options=(
                    SELECT u.DisplayName FROM users u
                    JOIN badges b ON u.Id = b.UserId
                    GROUP BY u.DisplayName 
                    HAVING COUNT(*) >= 200
                ),
                quantifier='{2}'
            )
        }}""",
        "Notes": "Subjective question - why is 'Glen_b' more based off of a real name than 'whuber'?",
    },
    {
        "Query ID": 106,
        "DB used": "codebase_community",
        "Query": "Of the top 5 users with the most views, who has their social media linked in their AboutMe section?",
        "(TAG baseline) Text2SQL Input": "List the top 5 users with the most views.",
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": "whuber",
        "BlendSQL": """WITH top_users AS (
            SELECT AboutMe, DisplayName FROM users 
            ORDER BY Views DESC LIMIT 5
        ) SELECT DisplayName FROM top_users
        WHERE {{LLMMap('Is a social media link present in this text?', 'top_users::AboutMe')}} = TRUE
        """,
        "Notes": None,
    },
    {
        "Query ID": 107,
        "DB used": "codebase_community",
        "Query": "Of all the comments commented by the user with a username of Harvey Motulsky and with a score of 5, rank the post ids in order of most helpful to least helpful",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": ["89457", "64710", "4945"],
        "order_insensitive_answer": False,
        "BlendSQL": """WITH harvey_comments AS (
            SELECT c.PostId, c.Text FROM comments c 
            JOIN users u ON u.Id = c.UserId
            WHERE c.Score = 5
            AND u.DisplayName = 'Harvey Motulsky'
        ) SELECT * FROM VALUES {{
            LLMQA(
                'Which PostIds are attached to the 3 most helpful texts?',
                (
                    SELECT * FROM harvey_comments
                ),
                options='harvey_comments::PostId'
                quantifier='{3}'
            )
        }}""",
        "Notes": "Subjective - what is 'most helpful'?",
    },
    {
        "Query ID": 61,
        "DB used": "california_schools",
        "Query": "Of the cities containing exclusively virtual schools which are the top 3 safest places to live?",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": ["Thousand Oaks", "Simi Valley", "Westlake Village"],
        "order_insensitive_answer": True,
        "BlendSQL": """SELECT * FROM VALUES {{
            LLMQA(
                'Which 3 cities are considered the safest places to live?',
                options=(
                    SELECT DISTINCT City FROM schools
                    WHERE Virtual = 'F'
                )
            )
        }}""",
        "Notes": "Subjective question - 'safest place to live' by what standard?",
    },
    {
        "Query ID": 62,
        "DB used": "california_schools",
        "Query": "List the cities containing the top 5 most enrolled schools in order from most diverse to least diverse.",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": [
            "Long Beach",
            "Paramount",
            "Granada Hills",
            "Temecula",
            "Carmichael",
        ],
        "order_insensitive_answer": False,
        "BlendSQL": None,
        "Notes": "What does 'most diverse' mean?",
    },
    {
        "Query ID": 63,
        "DB used": "california_schools",
        "Query": "Please list the top three continuation schools with the lowest eligible free rates for students aged 5-17 and rank them based on the overall affordability of their respective cities.",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": [
            "Del Amigo High (Continuation)",
            "Rancho del Mar High (Continuation)",
            "Millennium High Alternative",
        ],
        "order_insensitive_answer": False,
        "BlendSQL": """WITH top_schools AS (
            SELECT 
                s.City, 
                s.School, 
                f."Free Meal Count (Ages 5-17)" / f."Enrollment (Ages 5-17)" AS frpm_rate
            FROM schools s 
            JOIN frpm f ON f.CDSCode = s.CDSCode 
            WHERE f."Educational Option Type" = 'Continuation School'
            AND frpm_rate IS NOT NULL
            ORDER BY frpm_rate ASC LIMIT 3
        ) SELECT * FROM VALUES {{
            LLMQA(
                'Rank the schools, from least affordable city to most affordable city.',
                context=(SELECT City, School FROM top_schools),
                options='top_schools::School'
            )
        }}""",
        "Notes": "Doesn't specify whether ranking should be increasing or decreasing",
    },
    {
        "Query ID": 64,
        "DB used": "california_schools",
        "Query": "Of the schools with the top 3 SAT excellence rate, which county of the schools has the strongest academic reputation?",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "Santa Clara",
        "BlendSQL": """WITH top_schools AS (
            SELECT DISTINCT s.County, 1.0 * ss."NumGE1500" / ss.NumTstTakr AS rate 
            FROM schools s 
            JOIN satscores ss ON s.CDSCode = ss.cds
            WHERE rate IS NOT NULL
            ORDER BY rate DESC LIMIT 3
        ) SELECT {{
            LLMQA(
                'Which county has the strongest academic reputation?',
                options='top_schools::County'
            )
        }}""",
        "Notes": "'Strongest academic reputations' is subjective - wouldn't Los Angeles be above Santa Clara?. Also, question asks for a ranked list, but gold answer (and written TAG program) returns the top.",
    },
    {
        "Query ID": 65,
        "DB used": "california_schools",
        "Query": "Among the cities with the top 10 lowest enrollment for students in grades 1 through 12, which are the top 2 most popular cities to visit?",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": ["Death Valley", "Shaver Lake"],
        "order_insensitive_answer": True,
        "BlendSQL": """WITH lowest_enrollment AS (
            SELECT s.City, SUM(f."Enrollment (K-12)") AS total_enrollment 
            FROM schools s 
            JOIN frpm f ON s.CDSCode = f.CDSCode 
            GROUP BY s.City 
            ORDER BY total_enrollment ASC 
            LIMIT 10
        ) SELECT * FROM VALUES {{
            LLMQA(
                'Which 2 California cities are the most popular to visit?',
                options='lowest_enrollment::City',
                quantifier='{2}'
            )
        }}""",
        "Notes": """'Most popular cities to visit' is subjective? But, looking at online resources, it also seems wrong. 
        Yosemite (where Wawona is) has 4 million visitors per year: https://www.nps.gov/yose/planyourvisit/traffic.htm#:~:text=Each%20year%2C%20Yosemite%20National%20Park,no%20lodging%20or%20campground%20availability.
        Shaver Lake has less information, but this resource estimates 200,000+ per year: https://www.sce.com/sites/default/files/inline-files/RecreationWorkshop.pdf
        """,
    },
    {
        "Query ID": 952,
        "DB used": "formula_1",
        "Query": "Of the constructors that have been ranked 1 in 2014, which has the most prestige?",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "Ferrari",
        "BlendSQL": """WITH top_constructors AS (
            SELECT DISTINCT c.name FROM constructors c 
            JOIN results r ON r.constructorId = c.constructorId
            JOIN races ra ON r.raceId = ra.raceId
            WHERE r.rank = 1 AND ra.year = 2014
        ) SELECT {{
            LLMQA(
                "Which company's logo looks the most like Secretariat?",
                options='top_constructors::name'
            )
        }}""",
        "Notes": "'Most prestige' is subjective. Also - in `hand_written.py`, this question is different: 'Of the constructors that have been ranked 1 in 2014, whose logo looks most like Secretariat?'",
    },
    {
        "Query ID": 1000,
        "DB used": "formula_1",
        "Query": "Of the 5 racetracks that hosted the most recent races, rank the locations by distance to the equator.",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Ranking",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": ["Mexico City", "Sao Paulo", "Abu Dhabi", "Austin", "Suzuka"],
        "order_insensitive_answer": False,
        "BlendSQL": """WITH recent_races AS (
            SELECT c.location FROM races ra 
            JOIN circuits c ON c.circuitId = ra.circuitId
            ORDER BY ra.date DESC LIMIT 5
        ) SELECT * FROM VALUES {{
            LLMQA(
                'Order the locations by distance to the equator (closest -> farthest)',
                options='recent_races::location'
            )
        }}""",
        "Notes": "Question doesn't specify ascending or descending.",
    },
    {
        "Query ID": 81,
        "DB used": "california_schools",
        "Query": "Summarize the qualities of the schools with an average score in Math under 600 in the SAT test and are exclusively virtual.",
        "(TAG baseline) Text2SQL Input": "List the schools with an average score in Math under 600 in the SAT test and are exclusively virtual.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 82,
        "DB used": "california_schools",
        "Query": "Summarize the qualities of the schools in Riverside which the average math score for SAT is greater than 400.",
        "(TAG baseline) Text2SQL Input": "List the schools in Riverside which the average math score for SAT is greater than 400.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 85,
        "DB used": "codebase_community",
        "Query": "Summarize common characteristics of the titles of the posts owned by the user csgillespie.",
        "(TAG baseline) Text2SQL Input": "List the titles of the posts owned by the user csgillespie.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 86,
        "DB used": "codebase_community",
        "Query": "What qualities are represented by the badges obtained by csgillespie?",
        "(TAG baseline) Text2SQL Input": "List by the badges obtained by csgillespie?",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 87,
        "DB used": "codebase_community",
        "Query": "What is the average sentiment of the posts owned by the user csgillespie, Positive or Negative?",
        "(TAG baseline) Text2SQL Input": "List the body of the posts owned by the user csgillespie, Positive or Negative?",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 88,
        "DB used": "codebase_community",
        "Query": "Summarize qualities of the comments made by user 'A Lion.'",
        "(TAG baseline) Text2SQL Input": "List the text of the comments made by user 'A Lion.'",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 89,
        "DB used": "codebase_community",
        "Query": "Summarize the comments made on the post titled 'How does gentle boosting differ from AdaBoost?' to answer the original question",
        "(TAG baseline) Text2SQL Input": "List the text of the comments of the post titled 'How does gentle boosting differ from AdaBoost?'",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 90,
        "DB used": "codebase_community",
        "Query": "Summarize the issues described in the comments on Neil McGuigan's posts.",
        "(TAG baseline) Text2SQL Input": "List the text of the comments on Neil McGuigan's posts.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 91,
        "DB used": "codebase_community",
        "Query": "Summarize the comments added to the post with the highest score.",
        "(TAG baseline) Text2SQL Input": "List the text of the comments on the post with the highest score.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 92,
        "DB used": "formula_1",
        "Query": "Summarize the track characteristics of the circuits in Germany.",
        "(TAG baseline) Text2SQL Input": "List the names of the circuits in Germany.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 93,
        "DB used": "formula_1",
        "Query": "Summarize the track characteristics of the circuits in Spain.",
        "(TAG baseline) Text2SQL Input": "List the names of the circuits in Spain.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 94,
        "DB used": "formula_1",
        "Query": "Provide information about the races held on Sepang International Circuit.",
        "(TAG baseline) Text2SQL Input": "List the names of races held on Sepang International Circuit.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 95,
        "DB used": "formula_1",
        "Query": "Summarize the qualities of all the drivers who finished the game in race No. 872.",
        "(TAG baseline) Text2SQL Input": "List the names of the drivers who finished the game in race No. 872.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 96,
        "DB used": "formula_1",
        "Query": "Summarize the track characteristiscs of the circuits in Lisbon, Portugal.",
        "(TAG baseline) Text2SQL Input": "List the names of the circuits in Lisbon, Portugal.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 97,
        "DB used": "formula_1",
        "Query": "Summarize the track characteristics of the US circuits.",
        "(TAG baseline) Text2SQL Input": "List the names of the US circuits.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 99,
        "DB used": "formula_1",
        "Query": "Summarize information about the French constructors that have a lap number of over 50.",
        "(TAG baseline) Text2SQL Input": "List French constructors that have a lap number of over 50.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 100,
        "DB used": "formula_1",
        "Query": "Summarize the track characteristics of the circuits in Italy.",
        "(TAG baseline) Text2SQL Input": "List the track names of the circuits in Italy.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 101,
        "DB used": "european_football_2",
        "Query": "Summarize attributes of the players with an overall rating of over 88 from 2008 to 2010?",
        "(TAG baseline) Text2SQL Input": "List players with an overall rating of over 88 from 2008 to 2010?",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 102,
        "DB used": "european_football_2",
        "Query": "Identify common characteristics of the players whose volley score and dribbling score are over 87.",
        "(TAG baseline) Text2SQL Input": "List the players whose volley score and dribbling score are over 87.",
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
    {
        "Query ID": 103,
        "DB used": "european_football_2",
        "Query": "Summarize attributes of the 10 heaviest players.",
        "(TAG baseline) Text2SQL Input": None,
        "Query type": "Aggregation",
        "Knowledge/Reasoning Type": "Reasoning",
        "Answer": None,
        "BlendSQL": None,
        "Notes": None,
    },
]
