import pandas as pd

from blendsql import BlendSQL
from blendsql.models import VLLM


if __name__ == "__main__":
    model = VLLM(
        model_name_or_path="RedHatAI/gemma-3-12b-it-quantized.w4a16",
        base_url="http://127.0.0.1:8000/v1/",
    )

    # First, extract player names
    bsql = BlendSQL(
        {
            "Players": pd.DataFrame(
                {
                    "player_name": [
                        "LeBron James",
                        "Stephen Curry",
                        "Kevin Durant",
                        "Giannis Antetokounmpo",
                        "Luka Doncic",
                        "Jayson Tatum",
                        "Joel Embiid",
                        "Nikola Jokic",
                        "Jimmy Butler",
                        "Kawhi Leonard",
                        "Paul George",
                        "Damian Lillard",
                        "Anthony Davis",
                        "Ja Morant",
                        "Devin Booker",
                        "Donovan Mitchell",
                        "Trae Young",
                        "Zion Williamson",
                        "Jaylen Brown",
                        "Tyler Herro",
                        "Bam Adebayo",
                        "Rudy Gobert",
                        "Karl-Anthony Towns",
                        "Domantas Sabonis",
                        "Pascal Siakam",
                        "Fred VanVleet",
                        "CJ McCollum",
                        "Tobias Harris",
                        "Russell Westbrook",
                        "Chris Paul",
                        "Klay Thompson",
                        "Draymond Green",
                        "Andrew Wiggins",
                        "Julius Randle",
                        "RJ Barrett",
                        "Scottie Barnes",
                        "Paolo Banchero",
                        "Chet Holmgren",
                        "Victor Wembanyama",
                        "Alperen Sengun",
                        "Evan Mobley",
                        "Jarrett Allen",
                        "Mikal Bridges",
                        "OG Anunoby",
                        "Jalen Green",
                        "Anthony Edwards",
                        "Jaden McDaniels",
                        "Desmond Bane",
                        "Tyrese Haliburton",
                        "De'Aaron Fox",
                    ]
                }
            ),
            "Teams": pd.DataFrame(
                {
                    "team_name": [
                        "Los Angeles Lakers",
                        "Golden State Warriors",
                        "Phoenix Suns",
                        "Milwaukee Bucks",
                        "Dallas Mavericks",
                        "Boston Celtics",
                        "Philadelphia 76ers",
                        "Denver Nuggets",
                        "Miami Heat",
                        "Los Angeles Clippers",
                        "Portland Trail Blazers",
                        "Memphis Grizzlies",
                        "New Orleans Pelicans",
                        "Atlanta Hawks",
                        "Minnesota Timberwolves",
                        "Sacramento Kings",
                        "Toronto Raptors",
                        "Cleveland Cavaliers",
                        "New York Knicks",
                        "Brooklyn Nets",
                        "Utah Jazz",
                        "San Antonio Spurs",
                        "Houston Rockets",
                        "Orlando Magic",
                        "Detroit Pistons",
                        "Chicago Bulls",
                        "Washington Wizards",
                        "Charlotte Hornets",
                        "Indiana Pacers",
                        "Oklahoma City Thunder",
                    ]
                }
            ),
        },
        model=model,
        verbose=True,
    )

    smoothie = bsql.execute(
        """
        SELECT player_name, team_name
        FROM Players p
        JOIN Teams t ON
        {{
            LLMJoin(
                p.player_name, 
                t.team_name, 
                join_criteria='The player was playing for the team in 2015.'
            )
        }}
        """,
    )
    smoothie.print_summary()
    print(smoothie.pl())
