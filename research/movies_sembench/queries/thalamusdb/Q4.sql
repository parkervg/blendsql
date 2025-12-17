select
cast(sum(case when NLfilter(reviewText, 'the review sentiment is positive')
then 1 else 0 end) as float) / count(*) as positivity_ratio
from Reviews
where id = 'taken_3'