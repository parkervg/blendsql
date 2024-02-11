1. the cases when blendsql outperforms gpt, what is the pattern?

The answer provided by blensql is more concise and more closely related to the table via the SQL-like syntax. Hence, as it is compared to answers of open-eneded questions, it tends to perform better for questions expecting more regulated contents from the heterogeneous table. 
For example questions that are expecting a concise numerical answer from the table.

The detailed observation level example is saved in `blendsql_win.csv`

2. when a mistake is made, do blendsql and gpt make the same mistakes?¶

Yes, judged by non-unitary denotation_acc, there are in total 
    
- 1,962 mistakes out of 3,466 total observation, which amount to a pct mistake rate of 0.57 
- 1,265 common mistakes out of 1,962 total mistakes, which amounts to a pct common mistake of 0.64

Those common mistakes are saved in `common_mistakes.csv`, an initial analysis found the error can be attributed to:
- text normlaization
    * numeridcal(4, 3460): 6th -> sixth, 3 -> three.
    * date(950): 18 August 1941 -> 1940-9-18.
    * unstructed lengthy answer(3): The address of the museum located in a Victorian House is 503 Peeples Street SW in the West End neighborhood of Atlanta, Georgia -> 503 Peeples Street SW

- wrong inference in the table?
    * Need to look up the original table, but found an error in 

3. what type of mistakes does Blendsql make, is it in the parser or the blender?
    * Need to look up the original table