# Define our few-shot examples for wikitq
blendsql_examples = [
    {
        "serialized_db": """
Table Description: Fabrice Santoro
CREATE TABLE w(
	row_id int,
	name text,
	2001 text,
	2002 text,
	2003 text,
	2004 text,
	2005 text,
	2006 text,
	2007 text,
	2008 text,
	2009 text,
	2010 text,
	career sr text,
	career win-loss text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	name	2001	2002	2003	2004	2005	2006	2007	2008	2009	2010	career sr	career win-loss
0	australian open	2r	1r	3r	2r	1r	qf	3r	2r	3r	1r	0 / 18	22–18
1	french open	4r	2r	2r	3r	1r	1r	1r	2r	1r	a	0 / 20	17–20
2	wimbledon	3r	2r	2r	2r	2r	2r	2r	1r	2r	a	0 / 14	11–14
*/
""",
        "question": "did he win more at the australian open or indian wells?",
        "blendsql": """SELECT name FROM w WHERE name IN ('australian open', 'indian wells') ORDER BY {{LLMMap('how many wins', 'w::career win-loss')}} DESC LIMIT 1;""",
    },
    {
        "serialized_db": """
Table Description: 2007 New Orleans Saints season
CREATE TABLE w(
	row_id int,
	week int,
	date text,
	opponent text,
	time text,
	game site text,
	tv text,
	result/score text,
	record text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	week	date	opponent	time	game site	tv	result/score	record
0	1	2007-9-6	indianapolis colts	t20:30 edt	rca dome	nbc	l 41 – 10	0–1
1	2	2007-9-16	tampa bay buccaneers	t13:0 edt	raymond james stadium	fox	l 31 – 14	0–2
2	3	2007-9-24	tennessee titans	t20:30 edt	louisiana superdome	espn	l 31 – 14	0–3
*/
""",
        "question": "what number of games were lost at home?",
        "blendsql": """SELECT COUNT(*) FROM w WHERE {{LLMMap('is it a loss?'; 'w::result/score')}} = TRUE AND {{LLMMap('is it the home court of New Orleans Saints?'; 'w::game site')}} = TRUE;""",
    },
    {
        "serialized_db": """
Table Description: 2007 New Orleans Saints season
CREATE TABLE 2007 w(
	row_id int,
	week int,
	date text,
	opponent text,
	time text,
	game site text,
	tv text,
	result/score text,
	record text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	week	date	opponent	time	game site	tv	result/score	record
0	1	2007-9-6	indianapolis colts	t20:30 edt	away	nbc	loss	0–1
1	2	2007-9-16	tampa bay buccaneers	t13:0 edt	home	fox	win	1-1
2	3	2007-9-24	tennessee titans	t20:30 edt	away	espn	loss	1-2
*/
""",
        "question": "what number of games were lost at home?",
        "blendsql": """SELECT COUNT(*) FROM w WHERE "result/score" = 'loss' AND "game site" = 'home'""",
    },
    {
        "serialized_db": """
Table Description: Demographics of Alaska
CREATE TABLE w(
	row_id int,
	by race text,
	white text,
	black text,
	aian* text,
	asian text,
	nhpi* text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	by race	white	black	aian*	asian	nhpi*
0	2000 (total population)	75.43%	4.46%	19.06%	5.24%	0.88%
1	2000 (hispanic only)	3.42%	0.33%	0.45%	0.16%	0.06%
2	2005 (total population)	74.71%	4.72%	18.77%	5.9%	0.88%
*/
""",
        "question": "which hispanic population had the greatest growth from 2000 to 2005?",
        "blendsql": """{{
                LLMQA(
                    'which race had the greatest value?', 
                    'SELECT * FROM w WHERE "by race" = "2000 (hispanic only)"',
                    options='white;black;aian*;asian;nhpi*'
                )
            }} 
         """,
    },
    {
        "serialized_db": """
Table Description: Highest mountain peaks of California
CREATE TABLE w(
	row_id int,
	rank int,
	mountain peak text,
	mountain range text,
	elevation text,
	prominence text,
	isolation text,
	location text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	rank	mountain peak	mountain range	elevation	prominence	isolation	location
0	1	mount whitney	sierra nevada	14505 ft; 4421 m	10080 ft; 3072 m	1646 mi; 2649 km	36°34′43″n 118°17′31″w﻿ / ﻿36.5786°n 118.292°w
1	2	mount williamson	sierra nevada	14379 ft; 4383 m	1677 ft; 511 m	5.4 mi; 8.7 km	36°39′21″n 118°18′40″w﻿ / ﻿36.6559°n 118.3111°w
2	3	white mountain peak	white mountains	14252 ft; 4344 m	7196 ft; 2193 m	67 mi; 109 km	37°38′3″n 118°15′21″w﻿ / ﻿37.6341°n 118.2557°w
*/
)""",
        "question": "which mountain peak has a prominence more than 10,000 ft?",
        "blendsql": """
        SELECT "mountain peak" FROM w WHERE {{LLMMap('prominence in ft?'; 'w::prominence')}} > 10000;
        """,
    },
    {
        "serialized_db": """
Table Description: Daegu FC
CREATE TABLE w(
	row_id int,
	season int,
	division int,
	tms. int,
	pos. int,
	fa cup text,
	afc cl text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	season	division	tms.	pos.	fa cup	afc cl
0	2003	1	12	11	quarter final	none
1	2004	1	13	10	round of 32	none
2	2005	1	13	8	quarter final	none
*/
""",
        "question": "how far did they make it in the fa cup after 2009?",
        "blendsql": """{{LLMQA('how far did they make?', 'SELECT "fa cup" FROM w WHERE season > 2009')}}""",
    },
    {
        "serialized_db": """
Table Description: Electricity in Sri Lanka
CREATE TABLE w(
	row_id int,
	filledcolumnname text,
	2005 int,
	2006 int,
	2007 int,
	2008 int,
	2009 int,
	2010 int,
	2011 int,
	2012 int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	filledcolumnname	2005	2006	2007	2008	2009	2010	2011	2012
0	hydro power	1293	1316	1326	1357	1379	1382	1401	1584
1	thermal	1155	1155	1155	1285	1290	1390	1690	1638
2	other renewables	3	3	3	3	15	45	50	90
*/
)""",
        "question": "did the hydro power increase or decrease from 2010 to 2012?",
        "blendsql": """
        SELECT CASE WHEN (SELECT "2010" FROM w WHERE filledcolumnname = 'hydro power') < (SELECT "2012" FROM w WHERE filledcolumnname = 'hydro power') THEN 'increase' ELSE 'decrease' END;
        """,
    },
    {
        "serialized_db": """
Table Description: List of political parties in Japan
CREATE TABLE w(
row_id int,
party text,
diet representation\nrepresentatives int,
diet representation\ncouncillors int,
party leader(s) text,
comments text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	party	diet representation\nrepresentatives	diet representation\ncouncillors	party leader(s)	comments
0	your party (yp); minna no tō みんなの党; ("everybody's party")	18	18	yoshimi watanabe reps.	conservative liberalism, neoliberalism, economic liberalism, libertarianism, anti-nuclear
1	japanese communist party (jcp); nihon kyōsan-tō 日本共産党	8	11	kazuo shii reps.	the japanese communist party is japan's oldest party. it was formed in 1922 as an underground organization in the empire of japan, but was legalized after world war ii during the occupation. it used to be a communist party, but the party has past_ref shifted to a socialist party.
2	people's life party (plp); seikatsu no tō 生活の党	7	2	ichirō ozawa reps.	life party was founded by ichirō ozawa and 14 other diet members who were in the 2022-7-4 party of japan after a leadership dispute between ozawa and yukiko kada.
*/
""",
        "question": "what party is listed previous to the new renaissance party?",
        "blendsql": """SELECT {{LLMMap('what is party name?', 'w::party')}} FROM w WHERE row_id = (SELECT row_id FROM w WHERE {{LLMMap('what is party name?', 'w::party')}} = 'new renaissance party') - 1;""",
    },
    {
        "serialized_db": """
Table Description: FC Seoul in Asian football
CREATE TABLE w(
	row_id int,
	# int,
	season int,
	competition text,
	date text,
	round text,
	opponent text,
	h / a text,
	result text,
	scorer (s) text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	#	season	competition	date	round	opponent	h / a	result	scorer (s)
0	35	2011	afc; champions league	2011-03-02 00:00:00	group stage	al-ain	a	1–0	s : dejan damjanović
1	36	2011	afc; champions league	2011-03-15 00:00:00	group stage	hangzhou greentown	h	3–0	s : dejan damjanović, ou kyoung-jun, mauricio molina
2	37	2011	afc; champions league	2011-04-06 00:00:00	group stage	nagoya grampus	a	1–1	s : choi hyun-tae; n : kensuke nagai
*/
""",
        "question": "how many consecutive games did dejan damjanovic score a goal in during the 2013 season?",
        "blendsql": """{{LLMQA('how many consecutive games did dejan damjanovic score a goal?', 'SELECT "scorer (s)" FROM w WHERE season = 2013')}}""",
    },
    {
        "serialized_db": """
Table Description: Electoral district of Lachlan
CREATE TABLE w(
	row_id int,
	member text,
	party text,
	term text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	member	party	term
0	john ryan	none	1859–1864
1	james martin	none	1864–1869
2	james watson	none	1869–1880
*/
""",
        "question": "of the members of the third incarnation of the lachlan, who served the longest?",
        "blendsql": """SELECT member FROM w ORDER BY {{LLMMap('how long did it last?', 'w::term')}} DESC LIMIT 1;""",
    },
    {
        "serialized_db": """
Table Description: Portugal in the Eurovision Song Contest 1979
CREATE TABLE w(
	row_id int,
	draw int,
	artist text,
	song text,
	points int,
	place text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	draw	artist	song	points	place
0	1	gonzaga coutinho	"tema para um homem só"	102	5th
1	2	pedro osório s.a.r.l.	"uma canção comercial"	123	3rd
2	3	concha	"qualquer dia, quem diria"	78	6th
*/
""",
        "question": "who was the last draw?",
        "blendsql": """SELECT "artist" FROM w ORDER by "draw" desc LIMIT 1;""",
    },
    {
        "serialized_db": """
Table Description: GER Class N31
CREATE TABLE w(
	row_id int,
	year int,
	order text,
	quantity int,
	ger nos. text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	year	order	quantity	ger nos.
0	1893	n31	1	999
1	1893	h33	10	979
2	1894	l33	10	989
*/
""",
        "question": "which had more ger numbers, 1898 or 1893?",
        "blendsql": """SELECT "year" FROM w WHERE "year" IN ('1898', '1893') GROUP by "year" ORDER by SUM ("ger nos.") desc LIMIT 1;""",
    },
    {
        "serialized_db": """
Table Description: List of spans
CREATE TABLE w(
	row_id int,
	tramway text,
	country text,
	city text,
	height of pylons text,
	span width,\nleaning straight line text,
	span width,\nhorizontal measurement text,
	height of cable over ground text,
	year of inauguration text,
	notes text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	tramway	country	city	height of pylons	span width,\nleaning straight line	span width,\nhorizontal measurement	height of cable over ground	year of inauguration	notes
0	peak 2 peak gondola	canada	whistler	65m	3024 m	3019 m	436 m	2008	3s aerial tramway constructed by doppelmayr
1	hut of regensburg material transport aerial railway	austria	falbeson	?	?	?	430 m	?	none
2	vanoise express	france	vanoise	none	1850 m	1800 m	380 m	2003	none
*/
""",
        "question": "was the sandia peak tramway innagurate before or after the 3s aerial tramway?",
        "blendsql": """SELECT (SELECT "year of inauguration" FROM w WHERE "tramway" = 'sandia peak tramway') < (SELECT "year of inauguration" FROM w WHERE "tramway" = '3s aerial tramway');""",
    },
    {
        "serialized_db": """
Table Description: Płock Governorate
CREATE TABLE w(
	row_id int,
	language text,
	number int,
	percentage (%) text,
	males int,
	females int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	language	number	percentage (%)	males	females
0	polish	447685	80.86	216794	230891
1	yiddish	51215	9.25	24538	26677
2	german	35931	6.49	17409	18522
*/
""",
        "question": "how many male and female german speakers are there?",
        "blendsql": """SELECT "males" + "females" FROM w WHERE "language" = 'german';""",
    },
]


sql_examples = [
    {
        "serialized_db": """
Table Description: Fabrice Santoro
CREATE TABLE w(
	row_id int,
	name text,
	2001 text,
	2002 text,
	2003 text,
	2004 text,
	2005 text,
	2006 text,
	2007 text,
	2008 text,
	2009 text,
	2010 text,
	career\nsr text,
	wins int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	name	2001	2002	2003	2004	2005	2006	2007	2008	2009	2010	career\nsr	wins
0	australian open	2r	1r	3r	2r	1r	qf	3r	2r	3r	1r	0 / 18	22
1	french open	4r	2r	2r	3r	1r	1r	1r	2r	1r	a	0 / 20	17
2	wimbledon	3r	2r	2r	2r	2r	2r	2r	1r	2r	a	0 / 14	11
*/
""",
        "question": "did he win more at the australian open or indian wells?",
        "sql": """SELECT name FROM w WHERE name IN ('australian open', 'indian wells') ORDER BY wins DESC LIMIT 1;""",
    },
    {
        "serialized_db": """
Table Description: 2007 New Orleans Saints season
CREATE TABLE w(
	row_id int,
	week int,
	date text,
	opponent text,
	time text,
	game site text,
	tv text,
	result text,
	record text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	week	date	opponent	time	game site	tv	result	record
0	1	2007-9-6	indianapolis colts	t20:30 edt	rca dome	nbc	l	0–1
1	2	2007-9-16	tampa bay buccaneers	t13:0 edt	raymond james stadium	fox	l	0–2
2	3	2007-9-24	tennessee titans	t20:30 edt	louisiana superdome	espn	l	0–3
*/
""",
        "question": "what number of games were lost at home?",
        "sql": """SELECT COUNT(*) FROM w WHERE result = 'l' AND "game site" = 'louisiana superdome';""",
    },
    {
        "serialized_db": """
Table Description: 2007 New Orleans Saints season
CREATE TABLE w(
	row_id int,
	week int,
	date text,
	opponent text,
	time text,
	game site text,
	tv text,
	result/score text,
	record text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	week	date	opponent	time	game site	tv	result/score	record
0	1	2007-9-6	indianapolis colts	t20:30 edt	away	nbc	loss	0–1
1	2	2007-9-16	tampa bay buccaneers	t13:0 edt	home	fox	win	1-1
2	3	2007-9-24	tennessee titans	t20:30 edt	away	espn	loss	1-2
*/
""",
        "question": "what number of games were lost at home?",
        "sql": """SELECT COUNT(*) FROM w WHERE "result/score" = 'loss' AND "game site" = 'home';""",
    },
    {
        "serialized_db": """
Table Description: Electricity in Sri Lanka
CREATE TABLE w(
	row_id int,
	filledcolumnname text,
	2005 int,
	2006 int,
	2007 int,
	2008 int,
	2009 int,
	2010 int,
	2011 int,
	2012 int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	filledcolumnname	2005	2006	2007	2008	2009	2010	2011	2012
0	hydro power	1293	1316	1326	1357	1379	1382	1401	1584
1	thermal	1155	1155	1155	1285	1290	1390	1690	1638
2	other renewables	3	3	3	3	15	45	50	90
*/
""",
        "question": "did the hydro power increase or decrease from 2010 to 2012?",
        "sql": """SELECT CASE WHEN (SELECT "2010" FROM w WHERE filledcolumnname = 'hydro power') < (SELECT "2012" FROM w WHERE filledcolumnname = 'hydro power') THEN 'increase' ELSE 'decrease' END""",
    },
    {
        "serialized_db": """
Table Description: Portugal in the Eurovision Song Contest 1979
CREATE TABLE w(
	row_id int,
	draw int,
	artist text,
	song text,
	points int,
	place text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	draw	artist	song	points	place
0	1	gonzaga coutinho	"tema para um homem só"	102	5th
1	2	pedro osório s.a.r.l.	"uma canção comercial"	123	3rd
2	3	concha	"qualquer dia, quem diria"	78	6th
*/
""",
        "question": "who was the last draw?",
        "sql": """SELECT "artist" FROM w ORDER by "draw" desc LIMIT 1""",
    },
    {
        "serialized_db": """
Table Description: GER Class N31
CREATE TABLE w(
	row_id int,
	year int,
	order text,
	quantity int,
	ger nos. text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	year	order	quantity	ger nos.
0	1893	n31	1	999
1	1893	h33	10	979
2	1894	l33	10	989
*/
""",
        "question": "which had more ger numbers, 1898 or 1893?",
        "sql": """SELECT "year" FROM w WHERE "year" IN ('1898','1893') GROUP by "year" ORDER by SUM ("ger nos.") desc LIMIT 1""",
    },
    {
        "serialized_db": """
Table Description: List of spans
CREATE TABLE w(
	row_id int,
	tramway text,
	country text,
	city text,
	height of pylons text,
	span width,\nleaning straight line text,
	span width,\nhorizontal measurement text,
	height of cable over ground text,
	year of inauguration text,
	notes text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	tramway	country	city	height of pylons	span width,\nleaning straight line	span width,\nhorizontal measurement	height of cable over ground	year of inauguration	notes
0	peak 2 peak gondola	canada	whistler	65m	3024 m	3019 m	436 m	2008	3s aerial tramway constructed by doppelmayr
1	hut of regensburg material transport aerial railway	austria	falbeson	?	?	?	430 m	?	none
2	vanoise express	france	vanoise	none	1850 m	1800 m	380 m	2003	none
*/
""",
        "question": "was the sandia peak tramway innagurate before or after the 3s aerial tramway?",
        "sql": """SELECT (SELECT "year of inauguration" FROM w WHERE "tramway" = 'sandia peak tramway') < (SELECT "year of inauguration" FROM w WHERE "tramway" = '3s aerial tramway')""",
    },
    {
        "serialized_db": """
Table Description: World Artistic Gymnastics Championships – Women's floor
CREATE TABLE w(
	id int,
	year int,
	location text,
	gold text,
	silver text,
	bronze text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
id	year	location	gold	silver	bronze
0	1950	basel	helena rakoczy	tereza kočiš	stefania reindlova
1	1954	rome	tamara manina	eva bosáková	maria gorokovskaya
2	1958	moscow	eva bosáková	larisa latynina	keiko tanaka
*/
""",
        "question": "where were the championships held before the 1962 prague championships?",
        "sql": """SELECT "location" FROM w WHERE "year" < 1962 ORDER by "year" desc LIMIT 1""",
    },
    {
        "serialized_db": """
Table Description: WSL World Heavyweight Championship
CREATE TABLE w(
	id int,
	wrestler: text,
	times: text,
	date: text,
	location: text,
	notes: text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
id	wrestler:	times:	date:	location:	notes:
0	jonnie stewart	1	1996-6-6	rochester, minnesota	defeated larry gligorovich to win the awa superstars of wrestling world heavyweight championship.
1	king kong bundy	1	1999-3-31	oshkosh, wisconsin	later stripped of the title by owner dale gagne.
2	the patriot; (danny dominion)	1	2000-7-29	pine bluff, arkansas	defeated dale gagne in an impromptu match to win the title.
*/
""",
        "question": "when did steve corino win his first wsl title?",
        "sql": """SELECT "date:" FROM w WHERE "wrestler:" = 'steve corino' ORDER by "date:" LIMIT 1""",
    },
    {
        "serialized_db": """
Table Description: Płock Governorate
CREATE TABLE w(
	row_id int,
	language text,
	number int,
	percentage (%) text,
	males int,
	females int)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	language	number	percentage (%)	males	females
0	polish	447685	80.86	216794	230891
1	yiddish	51215	9.25	24538	26677
2	german	35931	6.49	17409	18522
*/
""",
        "question": "how many male and female german speakers are there?",
        "sql": """SELECT "males" + "females" FROM w WHERE "language" = 'german'""",
    },
    {
        "serialized_db": """
Table Description: Shikoku Pilgrimage
CREATE TABLE w(
	row_id int,
	no. int,
	temple text,
	honzon (main image) text,
	city/town/village text,
	prefecture text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	no.	temple	honzon (main image)	city/town/village	prefecture
0	1	ryōzen-ji (霊山寺)	shaka nyorai	naruto	tokushima prefecture
1	2	gokuraku-ji (極楽寺)	amida nyorai	naruto	tokushima prefecture
2	3	konsen-ji (金泉寺)	shaka nyorai	itano	tokushima prefecture
*/
""",
        "question": "what is the difference in the number of temples between imabari and matsuyama?",
        "sql": """SELECT abs((SELECT COUNT ("temple") FROM w WHERE "city/town/village" = 'imabari') - (SELECT COUNT ("temple") FROM w WHERE "city/town/village" = 'matsuyama'));""",
    },
    {
        "serialized_db": """
Table Description: Athletics at the 2001 Goodwill Games – Results
CREATE TABLE w(
	row_id int,
	rank real,
	name text,
	nationality text,
	time text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	rank	name	nationality	time
0	nan	brahim boulami	morocco	2022-07-17 08:17:43
1	nan	reuben kosgei	kenya	2022-07-17 08:18:37
2	nan	stephen cherono	kenya	2022-07-17 08:19:58
*/
""",
        "question": "what counties had the least participants for the race?",
        "sql": """SELECT "nationality" FROM w GROUP by "nationality" having COUNT("name") = (SELECT COUNT ("name") FROM w GROUP by "nationality" ORDER by COUNT ("name") asc LIMIT 1);""",
    },
    {
        "serialized_db": """
Table Description: Saint Helena, Ascension and Tristan da Cunha
CREATE TABLE w(
	row_id int,
	administrative\narea text,
	area\nkm2 real,
	area\nsq mi int,
	population int,
	administrative\ncentre text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	administrative\narea	area\nkm2	area\nsq mi	population	administrative\ncentre
0	saint helena	122.0	47	5809	jamestown
1	ascension island	91.0	35	1532	georgetown
2	tristan da cunha	184.0	71	388	edinburgh of the 7 seas
*/
""",
        "question": "is the are of saint helena more than that of nightingale island?",
        "sql": """SELECT (SELECT "area\\nkm2" FROM w WHERE "administrative\\narea" = 'saint helena') > (SELECT "area\\nkm2" FROM w WHERE "administrative\\narea" = 'nightingale island');""",
    },
    {
        "serialized_db": """
Table Description: The Boys (comics)
CREATE TABLE w(
	row_id int,
	# int,
	title text,
	tpb isbn text,
	tpb release date text,
	tpb page number int,
	collected material text)
/*
3 example rows:
SELECT * FROM w LIMIT 3;
row_id	#	title	tpb isbn	tpb release date	tpb page number	collected material
0	1	the name of the game	isbn 91-33-30546-3	2007-06-01 00:00:00	152	the boys #1-6
1	2	get some	isbn 1-933305-68-1	2008-03-01 00:00:00	192	the boys #7–14
2	3	good for the soul	isbn 1-933305-92-4	2008-10-01 00:00:00	192	the boys #15-22
*/
""",
        "question": """what title appears before "the self-preservation society"?""",
        "sql": """SELECT "title" FROM w WHERE row_id = (SELECT row_id FROM w WHERE "title" = 'the self-preservation society') - 1;""",
    },
]
