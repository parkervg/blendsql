select reviewId
from Reviews
where NLfilter(originalScore, 'The score, as a fraction, is greater than 0.5.')
AND originalScore IS NOT NULL AND originalScore LIKE '%/%' AND CAST(split_part(originalScore, '/', 1) AS FLOAT) / CAST(split_part(originalScore, '/', 2) AS FLOAT) <> 0.5
limit 5