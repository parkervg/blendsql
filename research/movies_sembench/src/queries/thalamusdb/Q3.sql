select count(*) as positive_review_cnt
from Reviews
where id = 'taken_3'
and NLfilter(reviewText, 'The movie review has a positive sentiment')