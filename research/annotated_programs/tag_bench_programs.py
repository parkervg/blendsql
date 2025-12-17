ANNOTATED_TAG_DATASET = [
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
                         AND {{LLMMap('Is this a county in the California Bay Area?'
                           , s.County)}} = TRUE""",
        "DuckDB": """SELECT COUNT(DISTINCT s.CDSCode)
                     FROM schools s
                    JOIN satscores sa ON s.CDSCode = sa.cds
                     WHERE sa.AvgScrMath > 560
                       AND LLMMapBool(
                            'Is this a county in the California Bay Area?',
                            s.County,
                            NULL,
                            NULL
                        ) = TRUE
                  """,
        "LOTUS": """
        def f():
            query = "Among the schools with the average score in Math over 560 in the SAT test, how many schools are in counties in the bay area?"
            answer = 71
            scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
            scores_df = scores_df[scores_df["AvgScrMath"] > 560]
            unique_counties_df = scores_df[["cname"]].drop_duplicates()
            bay_area_counties_df = unique_counties_df.sem_filter("{cname} is in the bay area")
            bay_area_counties = bay_area_counties_df["cname"].tolist()
        
            bay_area_schools_df = scores_df[scores_df["cname"].isin(bay_area_counties)]
            prediction = len(bay_area_schools_df)
            return prediction, answer
        """,
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
                       WHERE {{LLMMap('Is this county in Southern California?'
                           , s.County)}} = TRUE
                         AND ss.AvgScrRead IS NOT NULL
                       ORDER BY ss.AvgScrRead ASC
                           LIMIT 1""",
        "DuckDB": """SELECT s.Phone
                     FROM satscores ss
                              JOIN schools s ON ss.cds = s.CDSCode
                     WHERE LLMMapBool(
                                   'Is this county in Southern California?',
                                   s.County,
                                   NULL,
                                   NULL
                           ) = TRUE
                       AND ss.AvgScrRead IS NOT NULL
                     ORDER BY ss.AvgScrRead ASC LIMIT 1
                  """,
        "LOTUS": """
        def f():
            query = (
                "What is the telephone number for the school with the lowest average score in reading in a county in Southern California?"
            )
            answer = "(562) 944-0033"
            scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            unique_counties_df = scores_df[["cname"]].drop_duplicates()
            bay_area_counties_df = unique_counties_df.sem_filter("{cname} is in Southern California")
            bay_area_counties = bay_area_counties_df["cname"].tolist()
        
            scores_df = scores_df[scores_df["cname"].isin(bay_area_counties)]
            scores_df = scores_df.loc[[scores_df["AvgScrRead"].idxmin()]]
        
            merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
            prediction = merged_df.Phone.values[0]
            return prediction, answer
        """,
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
        "BlendSQL": """SELECT SUM(ss.NumTstTakr)
                       FROM satscores ss
                       JOIN schools s ON s.CDSCode = ss.cds
                       WHERE {{
                           LLMMap(
                               'What is the population of this California county? Give your best guess.', 
                               s.County
                            )
                       }} > 2000000""",
        "DuckDB": """SELECT SUM(ss.NumTstTakr)
                     FROM satscores ss
                     JOIN schools s ON s.CDSCode = ss.cds
                     WHERE LLMMapInt(
                        'What is the population of this California county? Give your best guess.',
                        s.County,
                        NULL,
                        NULL
                    ) > 2000000""",
        "LOTUS": """
        def f():
            query = "How many test takers are there at the school/s in a county with population over 2 million?"
            answer = 244742
            scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            unique_counties = pd.DataFrame(schools_df["County"].unique(), columns=["County"])
            unique_counties = unique_counties.sem_map(
                "What is the population of {County} in California? Answer with only the number without commas. Respond with your best guess."
            )
            counties_over_2m = set()
            for _, row in unique_counties.iterrows():
                try:
                    if int(re.findall(r"\d+", row._map)[-1]) > 2000000:
                        counties_over_2m.add(row.County)
                except:
                    pass
        
            schools_df = schools_df[schools_df["County"].isin(counties_over_2m)]
            merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
            prediction = int(merged_df["NumTstTakr"].sum())
            return prediction, answer
        """,
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
                       WHERE {{LLMMap('Is this county in Silicon Valley?'
                           , County)}} = TRUE
                         AND County IS NOT NULL
                       ORDER BY Longitude DESC
                           LIMIT 1""",
        "DuckDB": """SELECT GSoffered
                     FROM schools
                     WHERE LLMMapBool('Is this county in Silicon Valley?', County, NULL, NULL) = TRUE
                       AND County IS NOT NULL
                     ORDER BY Longitude DESC LIMIT 1""",
        "LOTUS": """
        def f():
            query = "What is the grade span offered in the school with the highest longitude in counties that are part of the 'Silicon Valley' region?"
            answer = "K-5"
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            silicon_valley_cities_df = schools_df[["County"]].drop_duplicates().dropna()
            silicon_valley_cities_df = silicon_valley_cities_df.sem_filter("{County} is in the Silicon Valley region")
            silicon_valley_cities = silicon_valley_cities_df["County"].tolist()
        
            silicon_valley_schools_df = schools_df[schools_df["County"].isin(silicon_valley_cities)]
            highest_longitude_school_df = silicon_valley_schools_df.nlargest(1, "Longitude")
            prediction = highest_longitude_school_df["GSoffered"].values[0]
            return prediction, answer
        """,
        "Notes": "This is changed to 'counties' in hand_written.py. If Sonoma is in the Bay Area (many sources consider it to be), this answer would be K-12.",
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
        "BlendSQL": """WITH top_names AS (SELECT AdmFName1 AS name, COUNT(AdmFName1) AS count
                       FROM schools
                       GROUP BY "AdmFName1"
                       ORDER BY count DESC
                           LIMIT 20
                           )
        SELECT name
        FROM top_names
        WHERE {{LLMMap('Is this a female name?', name)}} = TRUE
        ORDER BY count DESC LIMIT 2""",
        "DuckDB": """WITH top_names AS (SELECT AdmFName1 AS name, COUNT(AdmFName1) AS count
                     FROM schools
                     GROUP BY "AdmFName1"
                     ORDER BY count DESC
                         LIMIT 20
                         )
        SELECT name
        FROM top_names
        WHERE LLMMapBool('Is this a female name?', name, NULL, NULL) = TRUE
        ORDER BY count DESC LIMIT 2""",
        "LOTUS": """
        def f():
            query = "What are the two most common first names among the female school administrators?"
            answer = ["Jennifer", "Lisa"]
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
        
            schools_df = (
                schools_df.groupby("AdmFName1").size().reset_index(name="count").sort_values("count", ascending=False).head(20)
            )
            schools_df = schools_df.sem_filter("{AdmFName1} is a female first name")
            prediction = schools_df["AdmFName1"].tolist()[:2]
            return prediction, answer
        """,
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
                       FROM posts p
                                JOIN users u ON u.Id = p.OwnerUserId
                       WHERE u.DisplayName = 'csgillespie'
                         AND ParentId IS NULL
                         AND {{LLMMap('Does this post mention academic papers?'
                           , p.Body)}} = TRUE""",
        "DuckDB": """SELECT COUNT(*)
                     FROM posts p
                     JOIN users u ON u.Id = p.OwnerUserId
                     WHERE u.DisplayName = 'csgillespie'
                       AND ParentId IS NULL
                       AND LLMMapBool('Does this post mention academic papers?', p.Body, NULL, NULL) = TRUE""",
        "LOTUS": """
        def f():
            query = "Among the posts owned by csgillespie, how many of them are root posts and mention academic papers?"
            answer = 4
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            users_df = users_df[users_df["DisplayName"] == "csgillespie"]
            posts_df = posts_df[posts_df["ParentId"].isna()]
            merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId")
            merged_df = merged_df.sem_filter("{Body} mentions academic papers")
        
            prediction = len(merged_df)
            return prediction, answer
        """,
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
                         AND {{LLMMap('Is this text about statistics?'
                           , Text)}} = TRUE""",
        "DuckDB": """SELECT COUNT(*)
                     FROM comments
                     WHERE Score = 17
                       AND LLMMapBool('Is this text about statistics?', Text, NULL, NULL) = TRUE""",
        "LOTUS": """
        def f():
            query = "How many of the comments with a score of 17 are about statistics?"
            answer = 4
            comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
            comments_df = comments_df[comments_df["Score"] == 17]
            comments_df = comments_df.sem_filter("{Text} is about statistics")
            prediction = len(comments_df)
        
            return prediction, answer
        """,
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
                         AND {{LLMMap('Does this text discuss the R programming language?'
                           , Body)}} = TRUE""",
        "DuckDB": """SELECT COUNT(*)
                     FROM posts
                     WHERE ViewCount > 80000
                       AND LLMMapBool('Does this text discuss the R programming language?', Body, NULL, NULL) = TRUE""",
        "LOTUS": """
        def f():
            query = "Of the posts with views above 80000, how many discuss the R programming language?"
            answer = 3
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            posts_df = posts_df[posts_df["ViewCount"] > 80000]
            posts_df = posts_df.sem_filter("{Body} discusses the R programming language")
            prediction = len(posts_df)
        
            return prediction, answer
        """,
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
                       WHERE {{LLMMap('Is this a country in the Middle East?', c.country)}} = TRUE""",
        "DuckDB": """SELECT DISTINCT r.name
                     FROM races r
                              JOIN circuits c ON r.circuitId = c.circuitId
                     WHERE LLMMapBool('Is this a country in the Middle East?', c.country, NULL, NULL) = TRUE""",
        "LOTUS": """
        def f():
            query = "Please give the name of the race held on the circuits in the smallest country in the Middle East by land size."
            answer = "Bahrain Grand Prix"
            circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
            races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
            circuits_df = circuits_df.sem_filter("{country} is the smallest country in the Middle East by land size")
        
            merged_df = pd.merge(circuits_df, races_df, on="circuitId", suffixes=["_circuit", "_race"]).drop_duplicates(
                subset="name_race"
            )
            prediction = merged_df["name_race"].tolist()[0]
        
            return prediction, answer
        """,
        "Notes": "Is Europe in the Middle East? The ground truth says it is.",
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
                       WHERE ra.year = 2008
                         AND ra.name = 'Australian Grand Prix'
                         AND {{LLMMap('Is this nationality Asian?'
                           , d.nationality)}} = TRUE""",
        "DuckDB": """SELECT COUNT(DISTINCT d.driverId) AS asian_driver_count
                     FROM drivers d
                              JOIN results r ON d.driverId = r.driverId
                              JOIN races ra ON r.raceId = ra.raceId
                     WHERE ra.year = 2008
                       AND ra.name = 'Australian Grand Prix'
                       AND LLMMapBool('Is this nationality Asian?', d.nationality, NULL, NULL) = TRUE""",
        "LOTUS": """
        def f():
            query = "How many Asian drivers competed in the 2008 Australian Grand Prix?"
            answer = 2
        
            drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
            races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
            results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
            nationalities_df = drivers_df[["nationality"]].drop_duplicates()
            asian_nationalities_df = nationalities_df.sem_filter("{nationality} is Asian")
            asian_nationalities = asian_nationalities_df["nationality"].tolist()
        
            drivers_df = drivers_df[drivers_df["nationality"].isin(asian_nationalities)]
            races_df = races_df[(races_df["name"] == "Australian Grand Prix") & (races_df["year"] == 2008)]
            merged_df = pd.merge(pd.merge(races_df, results_df, on="raceId"), drivers_df, on="driverId")
            prediction = len(merged_df)
        
            return prediction, answer
        """,
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
        "BlendSQL": """SELECT preferred_foot
                       FROM Player p JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
                       WHERE player_name = {{
                           LLMQA(
                               "Which player has the most Ballon d'Or awards?"
                           )
                       }} LIMIT 1""",
        "DuckDB": """SELECT preferred_foot
                     FROM Player p
                              JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
                     WHERE player_name = LLMQAStr(
                             'Which player has the most Ballon d''Or awards?',
                             NULL,
                             (SELECT LIST(player_name) FROM Player),
                             NULL
                                         ) LIMIT 1""",
        "LOTUS": """
        def f():
            query = "What is the preferred foot when attacking of the player with the most Ballon d'Or awards of all time?"
            answer = "left"
            players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
            attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
            key_player = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is the first and last name of the player who has won the most Ballon d'Or awards of all time? Respond with only the name and no other words."}]).choices[0].message.content
            players_df = players_df[players_df["player_name"] == key_player]
            merged_df = pd.merge(players_df, attributes_df, on="player_api_id")
            merged_df = merged_df[["player_name", "preferred_foot"]]
            prediction = merged_df["preferred_foot"].values[0]
            return prediction, answer
        """,
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
        "BlendSQL": """
                    SELECT player_name
                    FROM Player
                    WHERE birthday LIKE '1970%'
                      AND {{LLMMap('Would someone born on this day be an Aquarius?'
                        , birthday)}} = TRUE
                    """,
        "DuckDB": """
                  SELECT player_name
                  FROM Player
                  WHERE birthday LIKE '1970%'
                    AND LLMMapBool('Would someone born on this day be an Aquarius?', birthday, NULL, NULL) = TRUE
                  """,
        "LOTUS": """
        def f():
            query = "List the football player with a birthyear of 1970 who is an Aquarius"
            answer = "Hans Vonk"
            players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
            players_df = players_df[players_df["birthday"].str.startswith("1970")]
            players_df = players_df.sem_filter("Someone born on {birthday} would be an Aquarius")
            prediction = players_df["player_name"].values[0]
            return prediction, answer
        """,
        #         "BlendSQL": """WITH DateRange AS (
        # SELECT * FROM VALUES {{LLMQA('What are the start and end date ranges for an Aquarius? Respond in MM-DD.', regex='\\d{2}-\\d{2}', quantifier='{2}')}}
        # )
        # SELECT player_name FROM Player
        # WHERE birthday LIKE '1970%'
        # AND strftime('%m-%d', birthday) >= (SELECT min(column1, column2) FROM DateRange)
        # AND strftime('%m-%d', birthday) <= (SELECT max(column1, column2) FROM DateRange)""",
        "Notes": "The second program is quicker since it relies on native SQLite comparisons. But for parity with the LOTUS program, we measure the map variant.",
    },
    {
        "Query ID": 19,
        "DB used": "european_football_2",
        "Query": "Please list the league from the country which is landlocked.",
        "(TAG baseline) Text2SQL Input": "Please list the unique leagues and the country they are from",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "Switzerland Super League",
        "BlendSQL": """SELECT l.name
                       FROM League l
                                JOIN Country c ON l.country_id = c.id
                       WHERE {{LLMMap('Is this country landlocked?', c.name)}} = TRUE""",
        "DuckDB": """SELECT l.name
                     FROM League l
                              JOIN Country c ON l.country_id = c.id
                     WHERE LLMMapBool('Is this country landlocked?', c.name, NULL, NULL) = TRUE""",
        "LOTUS": """
        def f():
            query = "Please list the league from the country which is landlocked."
            answer = "Switzerland Super League"
            leagues_df = pd.read_csv("../pandas_dfs/european_football_2/League.csv")
            countries_df = pd.read_csv("../pandas_dfs/european_football_2/Country.csv")
            countries_df = countries_df.sem_filter("{name} is landlocked")
            merged_df = pd.merge(
                leagues_df, countries_df, left_on="country_id", right_on="id", suffixes=["_league", "_country"]
            )
            prediction = merged_df["name_league"].values[0]
        
            return prediction, answer
        """,
        "Notes": None,
    },
    {
        "Query ID": 20,
        "DB used": "european_football_2",
        "Query": "How many matches in the 2008/2009 season were held in countries where French is an official language?",
        "(TAG baseline) Text2SQL Input": "List the matches from the 2008/2009 season and the countries they were held in",
        "Query type": "Match",
        "Knowledge/Reasoning Type": "Knowledge",
        "Answer": "866",
        "BlendSQL": """SELECT COUNT(*)
                       FROM "Match" m
                                JOIN Country c ON m.country_id = c.id
                       WHERE m.season = '2008/2009'
                         AND {{LLMMap('Is French an official language in this country?'
                           , c.name)}} = TRUE
                    """,
        "DuckDB": """SELECT COUNT(*)
                     FROM "Match" m
                              JOIN Country c ON m.country_id = c.id
                     WHERE m.season = '2008/2009'
                       AND LLMMapBool('Is French an official language in this country?', c.name, NULL, NULL) = TRUE
                  """,
        "LOTUS": """
        def f():
            query = "How many matches in the 2008/2009 season were held in countries where French is an official language?"
            answer = 866
            matches_df = pd.read_csv("../pandas_dfs/european_football_2/Match.csv")
            countries_df = pd.read_csv("../pandas_dfs/european_football_2/Country.csv")
            matches_df = matches_df[matches_df["season"] == "2008/2009"]
            countries_df = countries_df.sem_filter("{name} has French as an official language")
            merged_df = pd.merge(matches_df, countries_df, left_on="country_id", right_on="id")
            prediction = len(merged_df)
        
            return prediction, answer
        """,
        # "BlendSQL": """SELECT COUNT(*) FROM "Match" m
        # JOIN Country c ON m.country_id = c.id
        # WHERE m.season = '2008/2009'
        # AND c.name IN {{LLMQA('In which of these countries is French an official language?')}}
        # """,
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
        "BlendSQL": """WITH top_teams AS (SELECT DISTINCT team_long_name AS name, away_team_Goal
                                          FROM Team t
                                                   JOIN "Match" m ON t.team_api_id = m.away_team_api_id
                                          ORDER BY away_team_goal DESC LIMIT 3
                           )
        SELECT {{LLMQA('Which team has the most fans?', options =top_teams.name)}}""",
        "DuckDB": """WITH top_teams AS (SELECT DISTINCT team_long_name AS name, away_team_Goal
                                        FROM Team t
                                                 JOIN "Match" m ON t.team_api_id = m.away_team_api_id
                                        ORDER BY away_team_goal DESC LIMIT 3
                         )
        SELECT LLMQAStr('Which team has the most fans?', NULL, (SELECT LIST(name) FROM top_teams), NULL)""",
        "LOTUS": """
        def f():
            query = "Of the top three away teams that scored the most goals, which one has the most fans?"
            answer = "FC Barcelona"
            teams_df = pd.read_csv("../pandas_dfs/european_football_2/Team.csv")
            matches_df = pd.read_csv("../pandas_dfs/european_football_2/Match.csv")
        
            merged_df = pd.merge(matches_df, teams_df, left_on="away_team_api_id", right_on="team_api_id")
            merged_df = (
                merged_df.sort_values("away_team_goal", ascending=False).drop_duplicates(subset="team_long_name").head(3)
            )
            prediction = merged_df.sem_topk("What {team_long_name} has the most fans?", 1).team_long_name.values[0]
            return prediction, answer
        """,
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
        "BlendSQL": """SELECT ROUND(CAST(ym.Date AS INT) / 100)::INT AS "year"
                       FROM customers c
                                JOIN yearmonth ym ON c.CustomerID = ym.CustomerID
                       WHERE c.Currency = {{LLMQA('Which currency is the higher value?')}}
                       GROUP BY "year"
                       ORDER BY SUM (ym.Consumption) DESC LIMIT 1
                    """,
        "DuckDB": """SELECT CAST(ym.Date AS INT) / 100 AS "year"
                     FROM customers c
                              JOIN yearmonth ym ON c.CustomerID = ym.CustomerID
                     WHERE c.Currency =
                           LLMQAStr('Which currency is the higher value?', NULL, (SELECT LIST(Currency) FROM customers),
                                    NULL)
                     GROUP BY "year"
                     ORDER BY SUM(ym.Consumption) DESC LIMIT 1
                  """,
        "LOTUS": """
        def f():
            query = "Which year recorded the most gas use paid in the higher value currency?"
            answer = 2013
            customers_df = pd.read_csv("../pandas_dfs/debit_card_specializing/customers.csv")
            yearmonth_df = pd.read_csv("../pandas_dfs/debit_card_specializing/yearmonth.csv")
        
            unique_currencies = customers_df["Currency"].unique()
            most_value = (
                pd.DataFrame(unique_currencies, columns=["Currency"])
                .sem_topk("What {Currency} is the highest value currency?", 1)
                .Currency.values[0]
            )
            customers_df = customers_df[customers_df["Currency"] == most_value]
        
            yearmonth_df["year"] = yearmonth_df["Date"] // 100
            merged_df = pd.merge(customers_df, yearmonth_df, on="CustomerID")
            merged_df = merged_df.groupby("year")["Consumption"].sum().reset_index()
            merged_df = merged_df.sort_values("Consumption", ascending=False)
            prediction = int(merged_df["year"].values[0])
        
            return prediction, answer
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
        "BlendSQL": """SELECT {{LLMMap('Is the post relevant to Machine Learning?', p.Body, options =('YES', 'NO'))}}
                       FROM posts p JOIN votes v
                       ON p.Id = v.PostId
                       WHERE v.UserId = 1465
                    """,
        "DuckDB": """SELECT CASE
                                WHEN LLMMapBool('Is the post relevant to Machine Learning?', p.Body, NULL, NULL) = TRUE
                                    THEN 'YES'
                                ELSE 'NO' END
                     FROM posts p
                              JOIN votes v ON p.Id = v.PostId
                     WHERE v.UserId = 1465
                  """,
        "LOTUS": """
        def f():
            query = "Among the posts that were voted by user 1465, determine if the post is relevant to machine learning. Respond with YES if it is and NO if it is not."
            answer = ["YES", "YES", "YES"]
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            votes_df = pd.read_csv("../pandas_dfs/codebase_community/votes.csv")
            votes_df = votes_df[votes_df["UserId"] == 1465]
            merged_df = pd.merge(posts_df, votes_df, left_on="Id", right_on="PostId")
            merged_df = merged_df.sem_map(
                "{Body} is relevant to machine learning. Answer with YES if it is and NO if it is not."
            )
            prediction = merged_df._map.tolist()
            return prediction, answer
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
        "BlendSQL": """SELECT {{LLMMap('Extract the most statistical term from the title', p.Title, return_type='substring')}}
                       FROM posts p JOIN users u
                       ON p.OwnerUserId = u.Id
                       WHERE u.DisplayName = 'Vebjorn Ljosa'""",
        "DuckDB": """SELECT LLMMapSubstr('Extract the most statistical term from the title', p.Title, NULL, NULL)
                     FROM posts p
                              JOIN users u ON p.OwnerUserId = u.Id
                     WHERE u.DisplayName = 'Vebjorn Ljosa'""",
        "LOTUS": """
        def f():
            query = "Extract the statistical term from the post titles which were edited by Vebjorn Ljosa."
            answer = ["beta-binomial distribution", "AdaBoost", "SVM", "Kolmogorov-Smirnov statistic"]
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            merged_df = pd.merge(posts_df, users_df, left_on="OwnerUserId", right_on="Id")
            merged_df = merged_df[merged_df["DisplayName"] == "Vebjorn Ljosa"]
            merged_df = merged_df.sem_map("Extract the statistical term from {Title}. Respond with only the statistical term.")
            prediction = merged_df._map.tolist()
            return prediction, answer
        """,
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
        "BlendSQL": """WITH new_c AS
                                (SELECT c.Id, c.Text
                                 FROM comments c
                                          JOIN posts p ON p.Id = c.PostId
                                          JOIN users u ON p.OwnerUserId = u.Id
                                 WHERE p.Title = 'Analysing wind data with R'
                                 ORDER BY u.CreationDate LIMIT 5
                           )
        SELECT Id
        FROM new_c
        WHERE {{LLMMap('Does the comment have a positive sentiment?', Text)}} = TRUE""",
        "DuckDB": """WITH new_c AS
                              (SELECT c.Id, c.Text
                               FROM comments c
                                        JOIN posts p ON p.Id = c.PostId
                                        JOIN users u ON p.OwnerUserId = u.Id
                               WHERE p.Title = 'Analysing wind data with R'
                               ORDER BY u.CreationDate LIMIT 5
                         )
        SELECT Id
        FROM new_c
        WHERE LLMMapBool('Does the comment have a positive sentiment?', Text, NULL, NULL) = TRUE""",
        "LOTUS": """
        def f():
            query = "List the Comment Ids of the positive comments made by the top 5 newest users on the post with the title 'Analysing wind data with R'"
            answer = [11449]
            comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            posts_df = posts_df[posts_df["Title"] == "Analysing wind data with R"]
            merged_df = pd.merge(comments_df, posts_df, left_on="PostId", right_on="Id", suffixes=["_comment", "_post"])
            merged_df = pd.merge(merged_df, users_df, left_on="UserId", right_on="Id", suffixes=["_merged", "_user"])
            merged_df = merged_df.sort_values(by=["CreationDate_user"], ascending=False).head(5)
            merged_df = merged_df.sem_filter("The sentiment of {Text} is positive")
            prediction = merged_df.Id_comment.tolist()
            return prediction, answer
        """,
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
                options=('True', 'False')
            )
        }}
        """,
        "DuckDB": """SELECT CASE
                                WHEN
                                    LLMQABool(
                                            'Is the content in "Body" true or false?',
                                            (SELECT STRING_AGG(Body, '\n---\n')
                                             FROM posts p
                                                      JOIN tags t ON t.ExcerptPostId = p.Id
                                             WHERE t.TagName = 'bayesian'),
                                            NULL,
                                            NULL
                                    ) = TRUE THEN 'True'
                                ELSE 'False' END
                  """,
        "LOTUS": """
        def f():
            query = 'For the post from which the tag "bayesian" is excerpted from, identify whether the body of the post is True or False. Answer with True or False ONLY.'
            answer = "TRUE"
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            tags_df = pd.read_csv("../pandas_dfs/codebase_community/tags.csv")
            tags_df = tags_df[tags_df["TagName"] == "bayesian"]
            merged_df = pd.merge(tags_df, posts_df, left_on="ExcerptPostId", right_on="Id")
            prediction = merged_df.sem_map("Determine whether the content in {Body} is true. Respond with only TRUE or FALSE.")[
                "_map"
            ].values[0]
            return prediction, answer
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
        "BlendSQL": """SELECT ROUND(AVG(t.Price))::INT
                       FROM transactions_1k t
                                JOIN gasstations g ON g.GasStationID = t.GasStationID
                       WHERE g.Country IN {{LLMQA('What are abbreviations for the country historically known as Bohemia? If there are multiple possible abbreviations list them as a python list with quotes around each abbreviation.')}}
                    """,
        "DuckDB": """SELECT CAST(ROUND(AVG(t.Price)) AS INT)
                     FROM transactions_1k t
                              JOIN gasstations g ON g.GasStationID = t.GasStationID
                     WHERE g.Country IN LLMQAList('What are abbreviations for the country historically known as Bohemia? If there are multiple possible abbreviations list them as a python list with quotes around each abbreviation.', NULL, NULL, NULL)
                  """,
        "LOTUS": """
        def f():
            query = "What is the average total price of the transactions taken place in gas stations in the country which is historically known as Bohemia, to the nearest integer?"
            answer = "453"
            transactions_df = pd.read_csv("../pandas_dfs/debit_card_specializing/transactions_1k.csv")
            gasstations_df = pd.read_csv("../pandas_dfs/debit_card_specializing/gasstations.csv")
        
            countries = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "What are abbreviations for the country historically known as Bohemia? If there are multiple possible abbreivations list them as a python list with quotes around each abbreviation. Answer with ONLY the list in brackets."}]).choices[0].message.content
        
            try:
                countries = eval(countries)
            except:
                countries = [countries]
            
        
            gasstations_df = gasstations_df[gasstations_df["Country"].isin(countries)]
        
            merged_df = pd.merge(transactions_df, gasstations_df, on="GasStationID")
            prediction = round(merged_df["Price"].mean())
        
            return prediction, answer
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
        "BlendSQL": """SELECT DisplayName
                       FROM users u
                                JOIN badges b ON u.Id = b.UserId
                       WHERE b.Name = 'Supporter'
                         AND u.Location = {{LLMQA("What's the capital city of Austria?")}}
                       ORDER BY u.Age DESC LIMIT 1
                    """,
        "DuckDB": """SELECT DisplayName
                     FROM users u
                              JOIN badges b ON u.Id = b.UserId
                     WHERE b.Name = 'Supporter'
                       AND u.Location =
                           LLMQAStr('What''s the capital city of Austria?', NULL, (SELECT LIST(Location) FROM users),
                                    NULL)
                     ORDER BY u.Age DESC LIMIT 1
                  """,
        "LOTUS": """
        def f():
            query = (
                "List the username of the oldest user located in the capital city of Austria who obtained the Supporter badge."
            )
            answer = "ymihere"
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            badges_df = pd.read_csv("../pandas_dfs/codebase_community/badges.csv")
            merged_df = pd.merge(users_df, badges_df, left_on="Id", right_on="UserId")
            merged_df = merged_df[merged_df["Name"] == "Supporter"]
        
            location = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is the capital city of Austria? Respond with only the city name and no other words."}]).choices[0].message.content
            location = f"{location}, Austria"
            merged_df = merged_df[merged_df["Location"] == location]
            prediction = merged_df.sort_values(by=["Age"], ascending=False).DisplayName.values.tolist()[0]
        
            return prediction, answer
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
        "BlendSQL": """
        WITH gas_consumption AS (
            SELECT ym.Consumption, c.Currency
            FROM customers c
            JOIN yearmonth ym ON c.CustomerID = ym.CustomerID
            WHERE CAST(ym.Date AS INT) / 100 = 2012
        ) SELECT CAST(ROUND((SELECT SUM(Consumption) FROM gas_consumption g WHERE g.Currency = {{LLMQA('Currency code of Czech Republic?')}}) - 
        (SELECT SUM(Consumption) FROM gas_consumption g WHERE g.Currency = {{LLMQA('Currency code of European Union?')}})) AS INT)
        """,
        "DuckDB": """WITH gas_consumption AS (SELECT ym.Consumption, c.Currency
                                              FROM customers c
                                                       JOIN yearmonth ym ON c.CustomerID = ym.CustomerID
                                              WHERE CAST(ym.Date AS INT) / 100 = 2012)
                     SELECT CAST(ROUND((SELECT SUM(Consumption)
                                        FROM gas_consumption g
                                        WHERE g.Currency = LLMQAStr('Currency code of European Union?', NULL,
                                                                    (SELECT LIST(Currency) FROM gas_consumption),
                                                                    NULL)) -
                                       (SELECT SUM(Consumption)
                                        FROM gas_consumption g
                                        WHERE g.Currency = LLMQAStr('Currency code of European Union?', NULL,
                                                                    (SELECT LIST(Currency) FROM gas_consumption),
                                                                    NULL))) AS INT)
                  """,
        "LOTUS": """
        def f():
            query = "What is the difference in gas consumption between customers who pay using the currency of the Czech Republic and who pay the currency of European Union in 2012, to the nearest integer?"
            answer = 402524570
            customers_df = pd.read_csv("../pandas_dfs/debit_card_specializing/customers.csv")
            yearmonth_df = pd.read_csv("../pandas_dfs/debit_card_specializing/yearmonth.csv")
        
            countries = {"Area": ["Czech Republic", "European Union"]}
            countries_df = pd.DataFrame(countries)
            currency_df = countries_df.sem_map(
                "Given {Area}, return the 3 letter currency code for the area. Answer with the code ONLY.", suffix="currency"
            )
            currencies = currency_df["currency"].values.tolist()
        
            yearmonth_df = yearmonth_df[yearmonth_df["Date"] // 100 == 2012]
        
            merged_df = pd.merge(customers_df, yearmonth_df, on="CustomerID")
            first_df = merged_df[merged_df["Currency"] == currencies[0]]
            second_df = merged_df[merged_df["Currency"] == currencies[1]]
        
            prediction = round(first_df["Consumption"].sum() - second_df["Consumption"].sum())
        
            return prediction, answer
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
        "BlendSQL": """WITH sme_payers AS (SELECT *
                                           FROM customers c
                                           WHERE c.Segment = 'SME')
                       SELECT CASE
                                  WHEN (SELECT COUNT(*) FROM sme_payers p WHERE p.Currency = 'CZK') >
                                       (SELECT COUNT(*) FROM sme_payers p WHERE p.Currency = {{LLMQA('What is the 3 letter code for the second-largest reserved currency in the world?')}})
        THEN 'Yes' ELSE 'No'
        END
                    """,
        "DuckDB": """WITH sme_payers AS (SELECT *
                                         FROM customers c
                                         WHERE c.Segment = 'SME')
                     SELECT CASE
                                WHEN (SELECT COUNT(*) FROM sme_payers p WHERE p.Currency = 'CZK') >
                                     (SELECT COUNT(*)
                                      FROM sme_payers p
                                      WHERE p.Currency = LLMQAStr(
                                              'What is the 3 letter code for the second-largest reserved currency in the world?',
                                              NULL, (SELECT LIST(Currency) FROM sme_payers), NULL))
                                    THEN 'Yes'
                                ELSE 'No' END
                  """,
        "LOTUS": """
        def f():
            query = "Is it true that more SMEs pay in Czech koruna than in the second-largest reserved currency in the world?"
            answer = "Yes"
            customers_df = pd.read_csv("../pandas_dfs/debit_card_specializing/customers.csv")
        
            customers_df = customers_df[customers_df["Segment"] == "SME"]
        
            first_df = customers_df[customers_df["Currency"] == "CZK"]
        
            currency = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "What is the 3 letter code for the second-largest reserved currency in the world? Respond with only the 3 letter code and no other words."}]).choices[0].message.content
            second_df = customers_df[customers_df["Currency"] == currency]
        
            if len(first_df) > len(second_df):
                prediction = "Yes"
            else:
                prediction = "No"
        
            return prediction, answer
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
                    SELECT COUNT(*)
                    FROM schools s
                             JOIN satscores ss ON ss.cds = s.CDSCode
                    WHERE s.City = {{LLMQA('What is the name of the city that is the county seat of Lake County, California?')}}
                      AND ss.AvgScrMath + ss.AvgScrWrite + ss.AvgScrRead >= 1500
                    """,
        "DuckDB": """
                  SELECT COUNT(*)
                  FROM schools s
                           JOIN satscores ss ON ss.cds = s.CDSCode
                  WHERE s.City =
                        LLMQAstr('What is the name of the city that is the county seat of Lake County, California?',
                                 NULL, (SELECT LIST(City) FROM schools), NULL)
                    AND ss.AvgScrMath + ss.AvgScrWrite + ss.AvgScrRead >= 1500
                  """,
        "LOTUS": """
        def f():
            query = "What is the total number of schools whose total SAT scores are greater or equal to 1500 whose mailing city is the county seat of Lake County, California?"
            answer = 2
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
            city = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is the name of the city that is the county seat of Lake County, California? Respond with only the city name and no other words."}]).choices[0].message.content
            schools_df = schools_df[schools_df["City"] == city]
            scores_df["total"] = scores_df["AvgScrRead"] + scores_df["AvgScrMath"] + scores_df["AvgScrWrite"]
            scores_df = scores_df[scores_df["total"] >= 1500]
            merged_df = pd.merge(scores_df, schools_df, left_on="cds", right_on="CDSCode")
            prediction = len(merged_df)
        
            return prediction, answer
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
        "BlendSQL": """SELECT COUNT(DISTINCT d.driverId)
                       FROM drivers d
                       JOIN results r ON d.driverId = r.driverId
                       WHERE r.rank = 2
                       AND CAST(SUBSTR(CAST(d.dob AS STRING), 1, 4) AS NUMERIC) > {{LLMQA('What year did the Vietnam war end?', return_type='int', regex='\d{4}')}}
                    """,
        "DuckDB": """SELECT COUNT(DISTINCT d.driverId)
                     FROM drivers d
                              JOIN results r ON d.driverId = r.driverId
                     WHERE r.rank = 2
                       AND CAST(SUBSTR(CAST(d.dob AS STRING), 1, 4) AS NUMERIC) >
                           LLMQAInt('What year did the Vietnam war end?', NULL, NULL, NULL)
                  """,
        "LOTUS": """
        def f():
            query = "How many drivers born after the end of the Vietnam War have been ranked 2?"
            answer = 27
            drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
            results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
            results_df = results_df[results_df["rank"] == 2]
            vietnamyear = int(client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "What is the year of the end of the Vietnam war? Respond with only the year and no other words."}]).choices[0].message.content)
        
            drivers_df["birthyear"] = drivers_df.dropna(subset=["dob"])["dob"].str[:4].astype(int)
            drivers_df = drivers_df[drivers_df["birthyear"] > vietnamyear]
            merged_df = pd.merge(drivers_df, results_df, on="driverId")
            prediction = merged_df["driverId"].nunique()
        
            return prediction, answer
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
        "BlendSQL": """WITH gp_races AS (SELECT country
                                         FROM races r
                                                  JOIN circuits c ON c.circuitId = r.circuitId
                                         WHERE r.name = 'European Grand Prix')
                       SELECT CAST(ROUND(1.0 *
                                         (SELECT COUNT(*) FROM gp_races WHERE {{LLMMap('Does the Bundesliga happen here?', country)}} = TRUE) / 
        (SELECT COUNT(*) FROM gp_races) * 100) AS INT)
                    """,
        "DuckDB": """WITH gp_races AS (
        SELECT country
                                       FROM races r
                                        JOIN circuits c ON c.circuitId = r.circuitId
                                       WHERE r.name = 'European Grand Prix')
                     SELECT CAST(ROUND(1.0 *
                                       (SELECT COUNT(*)
                                        FROM gp_races
                                        WHERE LLMMapBool('Does the Bundesliga happen here?', country, NULL, NULL) =
                                              TRUE) /
                                       (SELECT COUNT(*) FROM gp_races) * 100) AS INT)
                  """,
        "LOTUS": """
        def f():
            query = "Among all European Grand Prix races, what is the percentage of the races were hosted in the country where the Bundesliga happens, to the nearest whole number?"
            answer = 52
            circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
            races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
            races_df = races_df[races_df["name"] == "European Grand Prix"]
            merged_df = pd.merge(circuits_df, races_df, on="circuitId")
            denom = len(merged_df)
            merged_df = merged_df.sem_filter("{country} is where the Bundesliga happens")
            numer = len(merged_df)
            prediction = int(numer * 100 / denom)
            return prediction, answer
        """,
        # "BlendSQL": """WITH gp_races AS (
        # SELECT country FROM races r
        # JOIN circuits c ON c.circuitId = r.circuitId
        # WHERE r.name = 'European Grand Prix'
        # ) SELECT CAST(ROUND(1.0 *
        # (SELECT COUNT(*) FROM gp_races WHERE gp_races.country = {{LLMQA('Where does the Bundesliga happen?')}}) /
        # (SELECT COUNT(*) FROM gp_races) * 100) AS INT)
        # """,
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
        "BlendSQL": """SELECT ROUND(AVG(pa.overall_rating::FLOAT))::INT
                       FROM Player_Attributes pa
                       JOIN Player p ON p.player_api_id = pa.player_api_id
                       WHERE CAST(p.height AS FLOAT) > 170
                         AND CAST(p.height AS FLOAT) < {{LLMQA('How tall was Michael Jordan in cm? Give your best guess.')}}
                         AND CAST (SUBSTR(pa.date, 1, 4) AS NUMERIC) BETWEEN 2010 AND 2015
                    """,
        "DuckDB": """SELECT CAST(ROUND(AVG(pa.overall_rating)) AS INT)
                     FROM Player_Attributes pa
                              JOIN Player p ON p.player_api_id = pa.player_api_id
                     WHERE p.height > 170
                       AND p.height <
                           LLMQAInt('How tall was Michael Jordan in cm? Give your best guess.', NULL, NULL, NULL)
                       AND CAST(SUBSTR(pa.date, 1, 4) AS NUMERIC) BETWEEN 2010 AND 2015
                  """,
        "LOTUS": """
        def f():
            query = "From 2010 to 2015, what was the average overall rating, rounded to the nearest integer, of players who are higher than 170 and shorter than Michael Jordan?"
            answer = 69
            jordan_df = pd.DataFrame({"Name": ["Michael Jordan"]})
            jordan_df = jordan_df.sem_map(
                "Given {Name}, provide the height in cm. Answer with ONLY the number to one decimal place.", suffix="height"
            )
            jordan_height = float(jordan_df["height"].values.tolist()[0])
        
            players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
            players_df = players_df[players_df["height"] > 170]
            players_df = players_df[players_df["height"] < jordan_height]
        
            attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
            attributes_df["year"] = attributes_df["date"].str[:4].astype(int)
            attributes_df = attributes_df[attributes_df["year"] >= 2010]
            attributes_df = attributes_df[attributes_df["year"] <= 2015]
        
            merged_df = pd.merge(players_df, attributes_df, on="player_api_id")
            prediction = round(merged_df["overall_rating"].mean())
        
            return prediction, answer
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
        "BlendSQL": """WITH gp_drivers AS (SELECT CONCAT(d.forename, ' ', d.surname) AS name, ra.year
                                           FROM drivers d
                                                    JOIN results r ON r.driverId = d.driverId
                                                    JOIN races ra ON r.raceId = ra.raceId
                                           WHERE ra.name = 'Australian Grand Prix'
                                             AND ra.year = 2008
                                             AND r.time IS NOT NULL)
                       SELECT COUNT(*)
                       FROM gp_drivers
                       WHERE gp_drivers.year > {{LLMMap('What year did this driver debut?', gp_drivers.name, regex='\d{4}')}}
                    """,
        "DuckDB": """WITH gp_drivers AS (SELECT CONCAT(d.forename, ' ', d.surname) AS name, ra.year
                                         FROM drivers d
                                                  JOIN results r ON r.driverId = d.driverId
                                                  JOIN races ra ON r.raceId = ra.raceId
                                         WHERE ra.name = 'Australian Grand Prix'
                                           AND ra.year = 2008
                                           AND r.time IS NOT NULL)
                     SELECT COUNT(*)
                     FROM gp_drivers
                     WHERE gp_drivers.year > LLMMapInt('What year did this driver debut?', gp_drivers.name, NULL, NULL)
                  """,
        "LOTUS": """
        def pipeline_38():
            query = "Among the drivers that finished the race in the 2008 Australian Grand Prix, how many debuted earlier than Lewis Hamilton?"
            answer = 3
            drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
            races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
            results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
        
            races_df = races_df[races_df["name"] == "Australian Grand Prix"]
            races_df = races_df[races_df["year"] == 2008]
            results_df = results_df[results_df["time"].notnull()]
            drivers_df = drivers_df.sem_map(
                "What year did driver {forename} {surname} debut in Formula 1? Answer with the year ONLY.", suffix="debut"
            )
            drivers_df["debut"] = pd.to_numeric(drivers_df["debut"], errors="coerce")
            drivers_df = drivers_df.dropna(subset=["debut"])
        
            merged_df = pd.merge(results_df, races_df, on="raceId").merge(drivers_df, on="driverId")
            merged_df = merged_df[merged_df["year"] > merged_df["debut"]]
        
            prediction = len(merged_df)
        
            return prediction, answer
        """,
        # "BlendSQL": """WITH gp_drivers AS (
        # SELECT CONCAT(d.forename, ' ', d.surname) AS name FROM drivers d
        # JOIN results r ON r.driverId = d.driverId
        # JOIN races ra ON r.raceId = ra.raceId
        # WHERE ra.name = 'Australian Grand Prix'
        # AND ra.year = 2008
        # ) SELECT COUNT(*) FROM gp_drivers
        # WHERE {{LLMMap('What year did this driver debut?', gp_drivers.name, return_type='int')}} > {{LLMQA('What year did Lewis Hamilton debut in F1?', return_type='int')}}
        # """,
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
        "BlendSQL": """SELECT COUNT(*)
                       FROM Player p
                       WHERE CAST(SUBSTR(birthday, 1, 4) AS NUMERIC) > {{LLMQA('What year did the 14th FIFA World Cup take place?')}}
                    """,
        "DuckDB": """SELECT COUNT(*)
                     FROM Player p
                     WHERE CAST(SUBSTR(birthday, 1, 4) AS NUMERIC) >
                           LLMQAInt('What year did the 14th FIFA World Cup take place?', NULL, NULL, NULL)
                  """,
        "LOTUS": """
        def f():
            query = "How many players were born after the year of the 14th FIFA World Cup?"
            answer = 3028
            players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
            wcyear = int(client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "Return the year of the 14th FIFA World Cup. Answer with the year ONLY."}]).choices[0].message.content)
        
            players_df["birthyear"] = players_df["birthday"].str[:4].astype(int)
            players_df = players_df[players_df["birthyear"] > wcyear]
            prediction = len(players_df)
        
            return prediction, answer
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
        "BlendSQL": """SELECT COUNT(DISTINCT p.player_api_id)
                       FROM Player p
                                JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
                       WHERE CAST(p.height AS FLOAT) >= 180
                         AND CAST(pa.volleys AS INT) > 70
                         AND CAST(p.height AS FLOAT) > {{LLMQA('How tall is Bill Clinton in centimeters?')}}
                    """,
        "DuckDB": """SELECT COUNT(DISTINCT p.player_api_id)
                     FROM Player p
                              JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
                     WHERE CAST(p.height AS FLOAT) >= 180
                       AND CAST(pa.volleys AS INT) > 70
                       AND CAST(p.height AS FLOAT) > LLMQAInt('How tall is Bill Clinton in centimeters?', NULL, NULL, NULL)
                  """,
        "LOTUS": """
        def f():
            query = "Among the players whose height is over 180, how many of them have a volley score of over 70 and are taller than Bill Clinton?"
            answer = 88
            players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
            attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
            players_df = players_df[players_df["height"] >= 180]
            attributes_df = attributes_df[attributes_df["volleys"] > 70]
            merged_df = pd.merge(players_df, attributes_df, on="player_api_id").drop_duplicates(subset="player_api_id")
        
            steph_height = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "How tall is Bill Clinton in centimeters? Box your final answer with \\boxed{just a number}."}]).choices[0].message.content
            steph_height = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", steph_height).group(1))
        
            merged_df = merged_df[merged_df["height"] > int(steph_height)]
            prediction = merged_df["player_api_id"].nunique()
        
            return prediction, answer
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
        "BlendSQL": """SELECT COUNT(DISTINCT f.CDSCode)
                       FROM frpm f
                                JOIN satscores ss ON ss.cds = f.CDSCode
                       WHERE f."Free Meal Count (K-12)" / f."Enrollment (K-12)" > 0.1
                         AND ss.AvgScrRead + ss.AvgScrMath >= {{LLMQA('What is the maximum possible SAT score?')}} - 300
                    """,
        "DuckDB": """SELECT COUNT(DISTINCT f.CDSCode)
                     FROM frpm f
                              JOIN satscores ss ON ss.cds = f.CDSCode
                     WHERE f."Free Meal Count (K-12)" / f."Enrollment (K-12)" > 0.1
                       AND ss.AvgScrRead + ss.AvgScrMath >=
                           LLMQAInt('What is the maximum possible SAT score?', NULL, NULL, NULL) - 300
                  """,
        "LOTUS": """
        def f():
            query = "Give the number of schools with the percent eligible for free meals in K-12 is more than 0.1 and test takers whose test score is greater than or equal to the score one hundred points less than the maximum."
            answer = 1
            scores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
            frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
            frpm_df = frpm_df[(frpm_df["Free Meal Count (K-12)"] / frpm_df["Enrollment (K-12)"]) > 0.1]
            max_score = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is the maximum SAT score?. Box your final answer with \\boxed{just a number}."}]).choices[0].message.content
            max_score = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", max_score).group(1))
            scores_df = scores_df[scores_df["AvgScrRead"] + scores_df["AvgScrMath"] >= max_score - 300]
            merged_df = pd.merge(scores_df, frpm_df, left_on="cds", right_on="CDSCode").drop_duplicates(subset="cds")
            prediction = len(merged_df)
        
            return prediction, answer
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
        "BlendSQL": """SELECT COUNT(DISTINCT s.CDSCode)
                       FROM frpm f
                                JOIN schools s ON s.CDSCode = f.CDSCode
                       WHERE (f."Enrollment (K-12)" - f."Enrollment (Ages 5-17)") > {{LLMQA('How many days are in April?')}}
                    """,
        "DuckDB": """SELECT COUNT(DISTINCT s.CDSCode)
                     FROM frpm f
                              JOIN schools s ON s.CDSCode = f.CDSCode
                     WHERE (f."Enrollment (K-12)" - f."Enrollment (Ages 5-17)") >
                           LLMQAInt('How many days are in April?', NULL, NULL, NULL)
                  """,
        "LOTUS": """
        def f():
            query = "How many schools have the difference in enrollements between K-12 and ages 5-17 as more than the number of days in April?"
            answer = 1236
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
            avg_class_size = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "What is the number of days in April? Box your final answer with \\boxed{just a number}."}]).choices[0].message.content
            avg_class_size = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", avg_class_size).group(1))
        
            frpm_df = frpm_df[frpm_df["Enrollment (K-12)"] - frpm_df["Enrollment (Ages 5-17)"] > avg_class_size]
            merged_df = pd.merge(schools_df, frpm_df, on="CDSCode").drop_duplicates(subset="CDSCode")
            prediction = merged_df["CDSCode"].nunique()
        
            return prediction, answer
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
        "BlendSQL": """SELECT COUNT(DISTINCT u.Id)
                       FROM users u
                       WHERE u.UpVotes > 100
                         AND u.Age > {{LLMQA('What is the median age in America? Give your best guess.')}}
                    """,
        "DuckDB": """SELECT COUNT(DISTINCT u.Id)
                     FROM users u
                     WHERE u.UpVotes > 100
                       AND
                         u.Age > LLMQAInt('What is the median age in America? Give your best guess.', NULL, NULL, NULL)
                  """,
        "LOTUS": """
        def f():
            query = "Among the users who have more than 100 upvotes, how many of them are older than the median age in America?"
            answer = 32
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            users_df = users_df[users_df["UpVotes"] > 100]
            median_age = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is the median age in America? Box your final answer with \\boxed{just a number}."}]).choices[0].message.content
            median_age = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", median_age).group(1))
            users_df = users_df[users_df["Age"] > median_age]
            prediction = len(users_df)
        
            return prediction, answer
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
        "BlendSQL": """SELECT DISTINCT p.player_name
                       FROM Player p
                       WHERE CAST(p.height AS FLOAT) > {{LLMQA('What is 6 foot 8 in centimeters?')}}
                    """,
        "DuckDB": """SELECT DISTINCT p.player_name
                     FROM Player p
                     WHERE p.height > LLMQAInt('What is 6 foot 8 in centimeters?', NULL, NULL, NULL)
                  """,
        "LOTUS": """
        def f():
            query = "Please list the player names taller than 6 foot 8?"
            answer = ["Kristof van Hout"]
            players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
            height = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is 6 foot 8 in centimeters? Box your final answer with \\boxed{just a number}."}]).choices[0].message.content
            height = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", height).group(1))
            players_df = players_df[players_df["height"] > height]
            prediction = players_df["player_name"].values.tolist()
            return prediction, answer
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
        "BlendSQL": """SELECT COUNT(*)
                       FROM Player p
                       WHERE p.player_name LIKE 'Adam%'
                         AND CAST(p.weight AS FLOAT) > {{LLMQA('What is 77.1kg in pounds?')}}
                    """,
        "DuckDB": """SELECT COUNT(*)
                     FROM Player p
                     WHERE p.player_name LIKE 'Adam%'
                       AND p.weight > LLMQAInt('What is 77.1kg in pounds?', NULL, NULL, NULL)
                  """,
        "LOTUS": """
        def f():
            query = "How many players whose first names are Adam and weigh more than 77.1kg?"
            answer = 24
            players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
            players_df = players_df[players_df["player_name"].str.startswith("Adam")]
            pounds = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is 77.1kg in pounds? Box your final answer with \\boxed{just a number}."}]).choices[0].message.content
            pounds = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", pounds).group(1))
            players_df = players_df[players_df["weight"] > pounds]
            prediction = len(players_df)
        
            return prediction, answer
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
        "BlendSQL": """SELECT player_name
                       FROM Player p
                       WHERE CAST(p.height AS FLOAT) > {{LLMQA('What is 5 foot 11 in centimeters?')}}
                       ORDER BY player_name LIMIT 3
                    """,
        "DuckDB": """SELECT player_name
                     FROM Player p
                     WHERE p.height > LLMQAInt('What is 5 foot 11 in centimeters?', NULL, NULL, NULL)
                     ORDER BY player_name LIMIT 3
                  """,
        "LOTUS": """
        def f():
            query = "Please provide the names of top three football players who are over 5 foot 11 tall in alphabetical order."
            answer = ["Aaron Appindangoye", "Aaron Galindo", "Aaron Hughes"]
            players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
            height = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "What is 5 foot 11 in centimeters? Box your final answer with \\boxed{just a number}."}]).choices[0].message.content
            height = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", height).group(1))
            players_df = players_df[players_df["height"] > height]
            players_df = players_df.sort_values("player_name")
            prediction = players_df["player_name"].head(3).values.tolist()
        
            return prediction, answer
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
        "BlendSQL": """SELECT COUNT(*)
                       FROM transactions_1k t
                                JOIN gasstations gs ON t.GasStationID = gs.GasStationID
                       WHERE gs.Country = 'CZE'
                         AND t.Price > {{LLMQA('What is 45 USD in CZK?')}}
                    """,
        "DuckDB": """SELECT COUNT(*)
                     FROM transactions_1k t
                              JOIN gasstations gs ON t.GasStationID = gs.GasStationID
                     WHERE gs.Country = 'CZE'
                       AND t.Price > LLMQAInt('What is 45 USD in CZK?', NULL, NULL, NULL)
                  """,
        "LOTUS": """
        def f():
            query = "How many transactions taken place in the gas station in the Czech Republic are with a price of over 45 US dollars?"
            answer = 56
            transactions_df = pd.read_csv("../pandas_dfs/debit_card_specializing/transactions_1k.csv")
            gasstations_df = pd.read_csv("../pandas_dfs/debit_card_specializing/gasstations.csv")
            gasstations_df = gasstations_df[gasstations_df["Country"] == "CZE"]
            merged_df = pd.merge(transactions_df, gasstations_df, on="GasStationID")
            price = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "What is 45 USD in CZK? Box your final answer with \\boxed{just a number}."}]).choices[0].message.content
            price = float(re.search(r"\\boxed\{(\d+(\.\d+)?)\}", price).group(1))
        
            merged_df = merged_df[merged_df["Price"] > price]
            prediction = len(merged_df)
        
            return prediction, answer
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
                'Which circuit is located closer to a capital city?', 
                options=('Silverstone Circuit', 'Hockenheimring', 'Hungaroring')
            )

        }}
        """,
        "DuckDB": """SELECT 
            LLMQAStr(
                'Which circuit is located closer to a capital city?', 
                NULL,
                ['Silverstone Circuit', 'Hockenheimring', 'Hungaroring'], 
                NULL
            )
        """,
        "LOTUS": """
        def pipeline_48():
            query = "Which of these circuits is located closer to a capital city, Silverstone Circuit, Hockenheimring or Hungaroring?"
            answer = "Hungaroring"
            circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
            circuits_df = circuits_df[circuits_df["name"].isin(["Silverstone Circuit", "Hockenheimring", "Hungaroring"])]
        
            prediction = circuits_df.sem_topk("What circuit, named {name} is located closer to a capital city?", 1).name.values[
                0
            ]
        
            return prediction, answer
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
        "BlendSQL": """SELECT ra.name
                       FROM drivers d
                                JOIN results r ON d.driverId = r.driverId
                                JOIN races ra ON ra.raceId = r.raceId
                       WHERE d.forename = 'Alex'
                         AND d.surname = 'Yoong'
                         AND r.position < {{LLMQA('How many starting positions are typically in an F1 race?')}} / 2
                    """,
        "DuckDB": """SELECT ra.name
                     FROM drivers d
                              JOIN results r ON d.driverId = r.driverId
                              JOIN races ra ON ra.raceId = r.raceId
                     WHERE d.forename = 'Alex'
                       AND d.surname = 'Yoong'
                       AND r.position <
                           LLMQAInt('How many starting positions are typically in an F1 race?', NULL, NULL, NULL) / 2
                  """,
        "LOTUS": """
        def f():
            query = "Which race was Alex Yoong in when he was earlier than top half of typical number of starting positions in a race?"
            answer = "Australian Grand Prix"
            drivers_df = pd.read_csv("../pandas_dfs/formula_1/drivers.csv")
            drivers_df = drivers_df[(drivers_df["forename"] == "Alex") & (drivers_df["surname"] == "Yoong")]
            results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
            races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
            merged_df = pd.merge(drivers_df, results_df, on="driverId").merge(races_df, on="raceId")
            half_point = int(client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "What is the half point of typical number of starting positions in a F1 race? Answer with only the number."}]).choices[0].message.content)
            prediction = merged_df[merged_df["position"] < half_point].name.values[0]
        
            return prediction, answer
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
        "DuckDB": """SELECT LLMQAStr(
                                    'Which school name sounds the most futuristic?',
                                    (SELECT STRING_AGG(s.School, '\n---\n')
                                     FROM schools s
                                              JOIN satscores ss ON s.CDSCode = ss.cds
                                     WHERE s.Magnet = TRUE
                                       AND ss.NumTstTakr > 500),
                                    NULL,
                                    NULL
                            )
                  """,
        "LOTUS": """
        def f():
            query = "Among the magnet schools with SAT test takers of over 500, which school name sounds most futuristic?"
            answer = "Polytechnic High"
        
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            schools_df = schools_df[schools_df["Magnet"] == 1]
            satscores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
            satscores_df = satscores_df[satscores_df["NumTstTakr"] > 500]
            merged_df = pd.merge(schools_df, satscores_df, left_on="CDSCode", right_on="cds")
            prediction = merged_df.sem_topk("What {School} sounds most futuristic?", 1).School.values[0]
        
            return prediction, answer
        """,
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
        "BlendSQL": """WITH top_posts AS (SELECT Title
                                          FROM posts p
                                          ORDER BY p.ViewCount DESC
                           LIMIT 5
                           )
        SELECT *
        FROM VALUES {{LLMQA('Order the article titles, from most technical to least technical', options=top_posts.Title)}}
                    """,
        "DuckDB": """WITH top_posts AS (SELECT Title
                                        FROM posts p
                                        ORDER BY p.ViewCount DESC
                         LIMIT 5
                         )
        SELECT UNNEST(LLMQAList('Order the article titles, from most technical to least technical', NULL,
                                (SELECT LIST(Title) FROM top_posts), NULL))
                  """,
        "LOTUS": """
        def f():
            query = "Of the 5 posts wih highest popularity, list their titles in order of most technical to least technical."
            answer = [
                "How to interpret and report eta squared / partial eta squared in statistically significant and non-significant analyses?",
                "How to interpret F- and p-value in ANOVA?",
                "What is the meaning of p values and t values in statistical tests?",
                "How to choose between Pearson and Spearman correlation?",
                "How do I get the number of rows of a data.frame in R?",
            ]
        
            posts_df = (
                pd.read_csv("../pandas_dfs/codebase_community/posts.csv").sort_values(by=["ViewCount"], ascending=False).head(5)
            )
        
            prediction = posts_df.sem_topk("What {Title} is most technical?", 5).Title.values.tolist()
        
            return prediction, answer
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
        "BlendSQL": """SELECT p.Id
                       FROM posts p
                                JOIN comments c ON p.Id = c.PostId
                       WHERE CAST(c.CreationDate AS STRING) LIKE '2014-09-14%'
                         AND {{LLMMap("Is the sentiment on this comment grateful?"
                           , c.Text)}} = TRUE
                       GROUP BY p.Id
                       ORDER BY COUNT (c.Id) DESC
                           LIMIT 2
                    """,
        "DuckDB": """SELECT p.Id
                     FROM posts p
                              JOIN comments c ON p.Id = c.PostId
                     WHERE CAST(c.CreationDate AS STRING) LIKE '2014-09-14%'
                       AND LLMMapBool('Is the sentiment on this comment grateful?', c.Text, NULL, NULL) = TRUE
                     GROUP BY p.Id
                     ORDER BY COUNT(c.Id) DESC LIMIT 2
                  """,
        "LOTUS": """
        def f():
            query = "What are the Post Ids of the top 2 posts in order of most grateful comments received on 9-14-2014"
            answer = [115372, 115254]
        
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
            comments_df = comments_df[comments_df["CreationDate"].str.startswith("2014-09-14")]
            merged_df = pd.merge(posts_df, comments_df, left_on="Id", right_on="PostId")
            merged_df = merged_df.sem_filter("The sentiment on {Text} is that of someone being grateful.")
            merged_df = merged_df.groupby("Id_x").size().sort_values(ascending=False)
            prediction = list(merged_df.index[:2])
        
            return prediction, answer
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
        "BlendSQL": """WITH top_post AS (SELECT p.Id
                                         FROM posts p
                                                  JOIN users u ON p.OwnerUserId = u.Id
                                         WHERE u.DisplayName = 'csgillespie'
                                         ORDER BY p.ViewCount DESC LIMIT 1
                           )
        SELECT {{
            LLMQA(
            'Which of these comments is the most sarcastic?', options =(
            SELECT c.Text FROM comments c
            JOIN top_post ON top_post.Id = c.PostId
            )
            )
            }}
                    """,
        "DuckDB": """WITH top_post AS (SELECT p.Id
                                       FROM posts p
                                                JOIN users u ON p.OwnerUserId = u.Id
                                       WHERE u.DisplayName = 'csgillespie'
                                       ORDER BY p.ViewCount DESC LIMIT 1
                         )
        SELECT LLMQAStr(
                       'Which of these comments is the most sarcastic?',
                       NULL,
                       (SELECT LIST(c.Text)
                        FROM comments c
                                 JOIN top_post ON top_post.Id = c.PostId),
                       NULL
               )
                  """,
        "LOTUS": """
        def f():
            query = "For the post owned by csgillespie with the highest popularity, what is the most sarcastic comment?"
            answer = "That pirates / global warming chart is clearly cooked up by conspiracy theorists - anyone can see they have deliberately plotted even spacing for unequal time periods to avoid showing the recent sharp increase in temperature as pirates are almost entirely wiped out. We all know that as temperatures rise it makes the rum evaporate and pirates cannot survive those conditions."
        
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            users_df = users_df[users_df["DisplayName"] == "csgillespie"]
            merged_df = (
                pd.merge(posts_df, users_df, left_on="OwnerUserId", right_on="Id")
                .sort_values(by=["ViewCount"], ascending=False)
                .head(1)
            )
        
            comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
            merged_df_with_comments = pd.merge(merged_df, comments_df, left_on="Id_x", right_on="PostId")
            prediction = merged_df_with_comments.sem_topk("What {Text} is most sarcastic?", 1).Text.values[0]
        
            return prediction, answer
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
        "BlendSQL": """WITH popular_tags AS (SELECT TagName
                                             FROM tags t
                                             ORDER BY t.Count DESC LIMIT 10
                           )
        SELECT {{
            LLMQA(
            'Which of these tags is LEAST related to statistics?', options =popular_tags.TagName
            )
            }}""",
        "DuckDB": """WITH popular_tags AS (SELECT TagName
                                           FROM tags t
                                           ORDER BY t.Count DESC LIMIT 10
                         )
        SELECT LLMQAStr(
                       'Which of these tags is LEAST related to statistics?',
                       NULL,
                       (SELECT LIST(TagName) FROM popular_tags),
                       NULL
               )""",
        "LOTUS": """
        def f():
            query = "Among the top 10 most popular tags, which is the least related to statistics?"
            answer = "self-study"
            tags_df = pd.read_csv("../pandas_dfs/codebase_community/tags.csv")
            tags_df = tags_df.sort_values("Count", ascending=False)
            tags_df = tags_df.head(10)
            prediction = tags_df.sem_topk("{TagName} is the least related to statistics?", 1).TagName.values[0]
            return prediction, answer
        """,
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
        "BlendSQL": """WITH favorited_posts AS (SELECT Id, Body
                                                FROM posts p
                                                ORDER BY p.FavoriteCount DESC LIMIT 10
                           )
        SELECT Id
        FROM favorited_posts
        WHERE Body = {{LLMQA('Which of these is the most lighthearted?')}}
                    """,
        "DuckDB": """WITH favorited_posts AS (SELECT Id, Body
                                              FROM posts p
                                              ORDER BY p.FavoriteCount DESC LIMIT 10
                         )
        SELECT Id
        FROM favorited_posts
        WHERE Body =
              LLMQAStr('Which of these is the most lighthearted?', NULL, (SELECT LIST(Body) FROM favorited_posts), NULL)
                  """,
        "LOTUS": """
        def f():
            query = "Of the top 10 most favorited posts, what is the Id of the most lighthearted post?"
            answer = 423
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            posts_df = posts_df.sort_values("FavoriteCount", ascending=False).head(10)
            prediction = posts_df.sem_topk("What {Body} is most lighthearted?", 1).Id.values[0]
            prediction = int(prediction)
        
            return prediction, answer
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
        "BlendSQL": """WITH filtered_posts AS (SELECT p.Id, p.Body
                                               FROM posts p
                                                        JOIN users u ON p.OwnerUserId = u.Id
                                               WHERE u.Age > 65
                                                 AND p.Score > 10)
                       SELECT *
                       FROM VALUES {{
            LLMQA(
                'Which 2 `Id` values are attached to the 2 posts whose authors have the least expertise?',
                context=(
                    SELECT * FROM filtered_posts
                ),
                options=filtered_posts.Id,
                quantifier='{2}'
            )
        }}
                    """,
        "DuckDB": """WITH filtered_posts AS (SELECT p.Id, p.Body
                                             FROM posts p
                                                      JOIN users u ON p.OwnerUserId = u.Id
                                             WHERE u.Age > 65
                                               AND p.Score > 10)
                     SELECT UNNEST(LLMQAList(
                             'Which 2 `Id` values are attached to the 2 posts whose authors have the least expertise?',
                             (SELECT STRING_AGG(Id || ' ' || Body, '\n---\n')
                              FROM posts p FROM filtered_posts ),
                filtered_posts.Id,
                '{2}'
            ))
                  """,
        "LOTUS": """
        def f():
            query = "Among the posts owned by a user over 65 with a score of over 10, what are the post id's of the top 2 posts made with the least expertise?"
            answer = [8485, 15670]
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            users_df = users_df[users_df["Age"] > 65]
            posts_df = posts_df[posts_df["Score"] > 10]
            merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId", suffixes=["_users", "_posts"])
            prediction = merged_df.sem_topk("What {Body} is made with the least expertise?", 2).Id_posts.values.tolist()
        
            return prediction, answer
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
        "BlendSQL": """WITH filtered_badges AS (SELECT b.Name
                                                FROM badges b
                                                         JOIN users u ON u.Id = b.UserId
                                                WHERE u.DisplayName = 'csgillespie')
                       SELECT {{
                           LLMQA(
                           'Which is most similar to an English grammar guide?', options =filtered_badges.Name
                           )
                           }}
                    """,
        "DuckDB": """WITH filtered_badges AS (SELECT b.Name
                                              FROM badges b
                                                       JOIN users u ON u.Id = b.UserId
                                              WHERE u.DisplayName = 'csgillespie')
                     SELECT LLMQA(
                                    'Which is most similar to an English grammar guide?',
                                    NULL,
                                    (SELECT LIST(Name) FROM filtered_badges),
                                    NULL
                            )
                  """,
        "LOTUS": """
        def f():
            query = "Among the badges obtained by csgillespie in 2011, which sounds most similar to an English grammar guide?"
            answer = "Strunk & White"
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            badges_df = pd.read_csv("../pandas_dfs/codebase_community/badges.csv")
            users_df = users_df[users_df["DisplayName"] == "csgillespie"]
            merged_df = pd.merge(users_df, badges_df, left_on="Id", right_on="UserId").drop_duplicates("Name")
            prediction = merged_df.sem_topk("What {Name} is most similar to an English grammar guide?", 1).Name.values[0]
        
            return prediction, answer
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
        "BlendSQL": """WITH yevgeny_posts AS (SELECT p.Id, p.Body
                                              FROM posts p
                                                       JOIN users u ON p.OwnerUserId = u.Id
                                              WHERE u.DisplayName = 'Yevgeny')
                       SELECT *
                       FROM VALUES {{
            LLMQA(
                'Which 2 `Id` values are attached to the 3 most pessimistic comments?',
                context=(
                    SELECT * FROM yevgeny_posts
                ),
                options=yevgeny_posts.Id,
                quantifier='{3}'
            )
        }}""",
        "DuckDB": """WITH yevgeny_posts AS (SELECT p.Id, p.Body
                                            FROM posts p
                                                     JOIN users u ON p.OwnerUserId = u.Id
                                            WHERE u.DisplayName = 'Yevgeny')
                     SELECT UNNEST(LLMQAList(
                             'Which 2 `Id` values are attached to the 3 most pessimistic comments?',
                             (SELECT STRING_AGG('Id: ' || Id || '\n' || 'Body: ' || Body, '\n---\n')
                              FROM yevgeny_posts),
                             (SELECT LIST(Id) FROM yevgeny_posts),
                             '{3}'
                                   ))""",
        "LOTUS": """
        def f():
            query = "Of the posts owned by Yevgeny, what are the id's of the top 3 most pessimistic?"
            answer = [23819, 24216, 35748]
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            posts_df = pd.read_csv("../pandas_dfs/codebase_community/posts.csv")
            users_df = users_df[users_df["DisplayName"] == "Yevgeny"]
            merged_df = pd.merge(users_df, posts_df, left_on="Id", right_on="OwnerUserId", suffixes=["_users", "_posts"])
            prediction = merged_df.sem_topk("What {Body} is most pessimistic?", 3).Id_posts.values.tolist()
        
            return prediction, answer
        """,
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
        "BlendSQL": """WITH top_players AS (SELECT p.player_name, AVG(CAST(pa.heading_accuracy AS INT)) AS avg_heading_accuracy
                                            FROM Player p JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
                                            WHERE CAST(p.height AS FLOAT) > 180
                                            GROUP BY p.player_api_id, p.player_name
                                            ORDER BY CAST(avg_heading_accuracy AS INT) DESC
                                            LIMIT 10
                           )
        SELECT *
        FROM VALUES {{
        LLMQA(
            "Which 3 of these names could be said to be the 'most unique'?", 
            options=top_players.player_name, 
            quantifier='{3}'
            )
        }}
                    """,
        "DuckDB": """WITH top_players AS (SELECT p.player_name, AVG(pa.heading_accuracy) AS avg_heading_accuracy
                                          FROM Player p
                                                   JOIN Player_Attributes pa ON p.player_api_id = pa.player_api_id
                                          WHERE p.height > 180
                                          GROUP BY p.player_api_id
                                          ORDER BY avg_heading_accuracy DESC
                         LIMIT 10
                         )
        SELECT UNNEST(LLMQAList(
                'Which 3 of these names could be said to be the ''most unique''?',
                NULL,
                (SELECT LIST(player_name) FROM top_players),
                '{3}'
                      ))
                  """,
        "LOTUS": """
        def f():
            query = "Of the top 10 players taller than 180 ordered by average heading accuracy, what are the top 3 most unique sounding names?"
            answer = ["Naldo", "Per Mertesacker", "Didier Drogba"]
            players_df = pd.read_csv("../pandas_dfs/european_football_2/Player.csv")
            attributes_df = pd.read_csv("../pandas_dfs/european_football_2/Player_Attributes.csv")
            players_df = players_df[players_df["height"] > 180]
            merged_df = pd.merge(players_df, attributes_df, on="player_api_id", suffixes=["_players", "_attributes"])
            merged_df = merged_df.groupby("player_api_id")
            merged_df = merged_df.agg({"heading_accuracy": "mean", "player_name": "first"}).reset_index()
            merged_df = merged_df.sort_values("heading_accuracy", ascending=False).head(10)
            prediction = merged_df.sem_topk("What {player_name} is most unique sounding?", 3).player_name.values.tolist()
        
            return prediction, answer
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
        "BlendSQL": """SELECT *
                       FROM VALUES {{
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
        "DuckDB": """SELECT UNNEST(LLMQAList(
                'Which 2 of these display names most based off of a real name?',
                NULL,
                (SELECT LIST(u.DisplayName)
                 FROM users u
                          JOIN badges b ON u.Id = b.UserId
                 GROUP BY u.DisplayName
                 HAVING COUNT(*) >= 200),
                '{2}'
                                   ))""",
        "LOTUS": """
        def f():
            query = "Out of users that have obtained at least 200 badges, what are the top 2 display names that seem most based off a real name?"
            answer = ["Jeromy Anglim", "Glen_b"]
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            badges_df = pd.read_csv("../pandas_dfs/codebase_community/badges.csv")
            merged_df = pd.merge(users_df, badges_df, left_on="Id", right_on="UserId")
            merged_df = merged_df.groupby("DisplayName").filter(lambda x: len(x) >= 200).drop_duplicates(subset="DisplayName")
            prediction = merged_df.sem_topk(
                "What {DisplayName} seems most based off a real name?", 2
            ).DisplayName.values.tolist()
            return prediction, answer
        """,
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
        "BlendSQL": """WITH top_users AS (SELECT AboutMe, DisplayName
                                          FROM users
                                          ORDER BY Views DESC LIMIT 5
                           )
        SELECT DisplayName
        FROM top_users
        WHERE {{LLMMap('Is a social media link present in this text?', AboutMe)}} = TRUE
                    """,
        "DuckDB": """WITH top_users AS (SELECT AboutMe, DisplayName
                                        FROM users
                                        ORDER BY Views DESC LIMIT 5
                         )
        SELECT DisplayName
        FROM top_users
        WHERE LLMMapBool('Is a social media link present in this text?', AboutMe, NULL, NULL) = TRUE
                  """,
        "LOTUS": """
        def f():
            query = "Of the top 5 users with the most views, who has their social media linked in their AboutMe section?"
            answer = "whuber"
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            users_df = users_df.sort_values("Views", ascending=False).head(5)
            prediction = users_df.sem_filter("The {AboutMe} contains a link to social media.").DisplayName.values[0]
            return prediction, answer
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
        "BlendSQL": """WITH harvey_comments AS (SELECT c.PostId, c.Text
                                                FROM comments c
                                                         JOIN users u ON u.Id = c.UserId
                                                WHERE c.Score = 5
                                                  AND u.DisplayName = 'Harvey Motulsky')
                       SELECT *
                       FROM VALUES {{
            LLMQA(
                'Rank the post IDs in order of most helpful to least helpful.',
                (
                    SELECT * FROM harvey_comments
                ),
                options=harvey_comments.PostId,
                quantifier='{3}'
            )
        }}""",
        "DuckDB": """WITH harvey_comments AS (SELECT c.PostId, c.Text
                                              FROM comments c
                                                       JOIN users u ON u.Id = c.UserId
                                              WHERE c.Score = 5
                                                AND u.DisplayName = 'Harvey Motulsky')
                     SELECT UNNEST(LLMQAList(
                             'Rank the post IDs in order of most helpful to least helpful.',
                             (SELECT STRING_AGG('PostId: ' || PostId || '\n' || 'Text: ' || Text, '\n---\n')
                              FROM harvey_comments),
                             (SELECT LIST(PostId)
                              FROM harvey_comments),
                             '{3}'
                                   ))""",
        "LOTUS": """
        def f():
            query = "Of all the comments commented by the user with a username of Harvey Motulsky and with a score of 5, rank the post ids in order of most helpful to least helpful."
            answer = [89457, 64710, 4945]
            users_df = pd.read_csv("../pandas_dfs/codebase_community/users.csv")
            comments_df = pd.read_csv("../pandas_dfs/codebase_community/comments.csv")
            users_df = users_df[users_df["DisplayName"] == "Harvey Motulsky"]
            comments_df = comments_df[comments_df["Score"] == 5]
            merged_df = pd.merge(users_df, comments_df, left_on="Id", right_on="UserId")
            prediction = merged_df.sem_topk("What {Text} is most helpful?", 3).PostId.values.tolist()
            return prediction, answer
        """,
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
        "BlendSQL": """SELECT *
                       FROM VALUES {{
            LLMQA(
                'Which 3 cities are considered the safest places to live?',
                options=(
                    SELECT DISTINCT City FROM schools
                    WHERE Virtual = 'F'
                ),
                quantifier='{3}'
            )
        }}""",
        "DuckDB": """SELECT UNNEST(LLMQAList(
                'Which 3 cities are considered the safest places to live?',
                NULL,
                (SELECT LIST(City)
                 FROM schools
                 WHERE Virtual = 'F'),
                '{3}'
                                   ))""",
        "LOTUS": """
        def f():
            query = "Of the cities containing exclusively virtual schools which are the top 3 safest places to live?"
            answer = ["Thousand Oaks", "Simi Valley", "Westlake Village"]
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            schools_df = schools_df[schools_df["Virtual"] == "F"].drop_duplicates(subset="City")
            prediction = schools_df.sem_topk("What {City} is the safest place to live?", 3).City.values.tolist()
            return prediction, answer
        """,
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
        "BlendSQL": """WITH top_schools AS (SELECT City
                                            FROM schools s
                                                     JOIN frpm f ON f.CDSCode = s.CDSCode
                                            ORDER BY f."Enrollment (K-12)" DESC LIMIT 5
                           )
        SELECT *
        FROM VALUES {{
            LLMQA(
                'Rank the cities, in order of most diverse to least diverse.', 
                options=top_schools.City,
                quantifier='{5}'
            )
        }}
                    """,
        "DuckDB": """WITH top_schools AS (SELECT City
                                          FROM schools s
                                                   JOIN frpm f ON f.CDSCode = s.CDSCode
                                          ORDER BY f."Enrollment (K-12)" DESC LIMIT 5
                         )
        SELECT UNNEST(LLMQAList(
                'Rank the cities, in order of most diverse to least diverse.',
                NULL,
                (SELECT LIST(City) FROM top_schools),
                '{5}'
                      ))
                  """,
        "LOTUS": """
        def f():
            query = "List the cities containing the top 5 most enrolled schools in order from most diverse to least diverse. "
            answer = ["Long Beach", "Paramount", "Granada Hills", "Temecula", "Carmichael"]
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
            merged_df = pd.merge(schools_df, frpm_df, on="CDSCode")
            merged_df = merged_df.sort_values("Enrollment (K-12)", ascending=False).head(5)
            prediction = merged_df.sem_topk("What {City} is the most diverse?", 5).City.values.tolist()
            return prediction, answer
        """,
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
        "BlendSQL": """WITH top_schools AS (SELECT s.City,
                                                   s.School,
                                                   f."Free Meal Count (Ages 5-17)" / f."Enrollment (Ages 5-17)" AS frpm_rate
                                            FROM schools s
                                                     JOIN frpm f ON f.CDSCode = s.CDSCode
                                            WHERE f."Educational Option Type" = 'Continuation School'
                                              AND frpm_rate IS NOT NULL
                                            ORDER BY frpm_rate ASC LIMIT 3
                           )
        SELECT *
        FROM VALUES {{
            LLMQA(
                'Rank the schools, from least affordable city to most affordable city.',
                context=(SELECT City, School FROM top_schools),
                options=top_schools.School,
                quantifier='{3}'
            )
        }}""",
        "DuckDB": """WITH top_schools AS (SELECT s.City,
                                                 s.School,
                                                 f."Free Meal Count (Ages 5-17)" / f."Enrollment (Ages 5-17)" AS frpm_rate
                                          FROM schools s
                                                   JOIN frpm f ON f.CDSCode = s.CDSCode
                                          WHERE f."Educational Option Type" = 'Continuation School'
                                            AND frpm_rate IS NOT NULL
                                          ORDER BY frpm_rate ASC LIMIT 3
                         )
        SELECT UNNEST(LLMQAList(
                'Rank the schools, from least affordable city to most affordable city.',
                (SELECT STRING_AGG('City: ' || City || '\n' || 'School: ' || School, '\n---\n') FROM top_schools),
                (SELECT LIST(School) FROM top_schools),
                '{3}'
                      ))""",
        "LOTUS": """
        def pipeline_63():
            query = "Please list the top three continuation schools with the lowest eligible free rates for students aged 5-17 and rank them based on the overall affordability of their respective cities."
            answer = ["Del Amigo High (Continuation)", "Rancho del Mar High (Continuation)", "Millennium High Alternative"]
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
            frpm_df = frpm_df[frpm_df["Educational Option Type"] == "Continuation School"]
            frpm_df["frpm_rate"] = frpm_df["Free Meal Count (Ages 5-17)"] / frpm_df["Enrollment (Ages 5-17)"]
            frpm_df = frpm_df.sort_values("frpm_rate", ascending=True).head(3)
            merged_df = pd.merge(schools_df, frpm_df, on="CDSCode")
            prediction = merged_df.sem_topk("What {City} is the most affordable?", 3).School.values.tolist()
            return prediction, answer
        """,
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
        "BlendSQL": """WITH top_schools AS (SELECT DISTINCT s.County, 1.0 * ss."NumGE1500" / ss.NumTstTakr AS rate
                                            FROM schools s
                                                     JOIN satscores ss ON s.CDSCode = ss.cds
                                            WHERE rate IS NOT NULL
                                            ORDER BY rate DESC LIMIT 3
                           )
        SELECT {{
            LLMQA(
            'Which county has the strongest academic reputation?', options =top_schools.County
            )
            }}""",
        "DuckDB": """WITH top_schools AS (SELECT DISTINCT s.County, 1.0 * ss."NumGE1500" / ss.NumTstTakr AS rate
                                          FROM schools s
                                                   JOIN satscores ss ON s.CDSCode = ss.cds
                                          WHERE rate IS NOT NULL
                                          ORDER BY rate DESC LIMIT 3
                         )
        SELECT LLMQAStr(
                       'Which county has the strongest academic reputation?',
                       NULL,
                       (SELECT LIST(County) FROM top_schools),
                       NULL
               )""",
        "LOTUS": """
        def f():
            query = "Of the schools with the top 3 SAT excellence rate, order their counties by academic reputation from strongest to weakest."
            answer = "Santa Clara"
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            satscores_df = pd.read_csv("../pandas_dfs/california_schools/satscores.csv")
            satscores_df["excellence_rate"] = satscores_df["NumGE1500"] / satscores_df["NumTstTakr"]
            satscores_df = satscores_df.sort_values("excellence_rate", ascending=False).head(3)
            merged_df = pd.merge(schools_df, satscores_df, left_on="CDSCode", right_on="cds")
            prediction = merged_df.sem_topk("What {County} has the strongest academic reputation?", 3).County.values[0]
            return prediction, answer
        """,
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
        "BlendSQL": """WITH lowest_enrollment AS (SELECT s.City, SUM(f."Enrollment (K-12)") AS total_enrollment
                                                  FROM schools s
                                                           JOIN frpm f ON s.CDSCode = f.CDSCode
                                                  GROUP BY s.City
                                                  ORDER BY total_enrollment ASC
                           LIMIT 10
                           )
        SELECT *
        FROM VALUES {{
            LLMQA(
                'Which 2 California cities are the most popular to visit?',
                options=lowest_enrollment.City,
                quantifier='{2}'
            )
        }}""",
        "DuckDB": """WITH lowest_enrollment AS (SELECT s.City, SUM(f."Enrollment (K-12)") AS total_enrollment
                                                FROM schools s
                                                         JOIN frpm f ON s.CDSCode = f.CDSCode
                                                GROUP BY s.City
                                                ORDER BY total_enrollment ASC
                         LIMIT 10
                         )
        SELECT UNNEST(LLMQAList(
                'Which 2 California cities are the most popular to visit?',
                NULL,
                (SELECT LIST(City) FROM lowest_enrollment),
                '{2}'
                      ))""",
        "LOTUS": """
        def pipeline_65():
            query = "Among the cities with the top 10 lowest enrollment for students in grades 1 through 12, which are the top 2 most popular cities to visit?"
            answer = ["Death Valley", "Shaver Lake"]
            schools_df = pd.read_csv("../pandas_dfs/california_schools/schools.csv")
            frpm_df = pd.read_csv("../pandas_dfs/california_schools/frpm.csv")
            merged_df = pd.merge(schools_df, frpm_df, on="CDSCode")
            merged_df = merged_df.groupby("City").agg({"Enrollment (K-12)": "sum"}).reset_index()
            merged_df = merged_df.sort_values("Enrollment (K-12)", ascending=True).head(10)
            prediction = merged_df.sem_topk(
                "What {City}/location in California is the most popular to visit?", 2
            ).City.values.tolist()
            return prediction, answer
        """,
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
        "BlendSQL": """WITH top_constructors AS (SELECT DISTINCT c.name
                                                 FROM constructors c
                                                          JOIN results r ON r.constructorId = c.constructorId
                                                          JOIN races ra ON r.raceId = ra.raceId
                                                 WHERE r.rank = 1
                                                   AND ra.year = 2014)
                       SELECT {{
                           LLMQA(
                           "Which company's logo looks the most like Secretariat?", options =top_constructors.name
                           )
                           }}""",
        "DuckDB": """WITH top_constructors AS (SELECT DISTINCT c.name
                                               FROM constructors c
                                                        JOIN results r ON r.constructorId = c.constructorId
                                                        JOIN races ra ON r.raceId = ra.raceId
                                               WHERE r.rank = 1
                                                 AND ra.year = 2014)
                     SELECT LLMQAStr(
                                    "Which company's logo looks the most like Secretariat?",
                                    NULL,
                                    (SELECT LIST(name) FROM top_constructors),
                                    NULL
                            )
                  """,
        "LOTUS": """
        def f():
            query = "Of the constructors that have been ranked 1 in 2014, whose logo looks most like Secretariat?"
            answer = "Ferrari"
            constructors_df = pd.read_csv("../pandas_dfs/formula_1/constructors.csv")
            results_df = pd.read_csv("../pandas_dfs/formula_1/results.csv")
            races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
            merged_df = pd.merge(results_df, constructors_df, on="constructorId", suffixes=["_results", "_constructors"])
            merged_df = pd.merge(merged_df, races_df, on="raceId", suffixes=["_merged", "_races"])
            merged_df = merged_df[(merged_df["rank"] == 1) & (merged_df["year"] == 2014)].drop_duplicates(
                subset="constructorId"
            )
            merged_df = merged_df.rename(columns={"name_merged": "name"})
            prediction = merged_df.sem_topk("What {name} logo is most like Secretariat?", 1).name.values[0]
            return prediction, answer
        """,
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
        "BlendSQL": """WITH recent_races AS (SELECT c.location
                                             FROM races ra
                                                      JOIN circuits c ON c.circuitId = ra.circuitId
                                             ORDER BY ra.date DESC LIMIT 5
                           )
        SELECT *
        FROM VALUES {{
            LLMQA(
                'Order the locations by distance to the equator (closest -> farthest)',
                options=recent_races.location,
                quantifier='{5}'
            )
        }}""",
        "DuckDB": """WITH recent_races AS (SELECT c.location
                                           FROM races ra
                                                    JOIN circuits c ON c.circuitId = ra.circuitId
                                           ORDER BY ra.date DESC LIMIT 5
                         )
        SELECT UNNEST(LLMQAList(
                'Order the locations by distance to the equator (closest -> farthest)',
                NULL,
                (SELECT LIST(location) FROM recent_races),
                '{5}'
                      ))""",
        "LOTUS": """
        def f():
            query = "Of the 5 racetracks that hosted the most recent races, rank the locations by distance to the equator."
            answer = ["Mexico City", "Sao Paulo", "Abu Dhabi", "Austin", "Suzuka"]
            circuits_df = pd.read_csv("../pandas_dfs/formula_1/circuits.csv")
            races_df = pd.read_csv("../pandas_dfs/formula_1/races.csv")
            merged_df = pd.merge(circuits_df, races_df, on="circuitId")
            merged_df = merged_df.sort_values("date", ascending=False).head(5)
            prediction = merged_df.sem_topk("What {location} is closest to the equator?", 5).location.values.tolist()
            return prediction, answer
        """,
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
