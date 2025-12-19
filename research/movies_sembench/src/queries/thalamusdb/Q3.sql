select count(*) as positive_review_cnt
from Reviews
where id = 'taken_3'
and NLfilter(reviewText, 'the review sentiment is positive')