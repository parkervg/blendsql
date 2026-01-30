select
cast(sum(case when NLfilter(originalScore, 'The score, as a fraction, is greater than 0.5.')
then 1 else 0 end) as float) / count(*) as positivity_ratio
from Reviews
where id = 'taken_3'
AND originalScore IS NOT NULL AND originalScore LIKE '%/%' AND CAST(split_part(originalScore, '/', 1) AS FLOAT) / CAST(split_part(originalScore, '/', 2) AS FLOAT) <> 0.5