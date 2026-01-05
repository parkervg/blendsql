select
cast(sum(case when NLfilter(reviewText, 'The movie review has a positive sentiment')
then 1 else 0 end) as float) / count(*) as positivity_ratio
from Reviews
where id = 'taken_3'