select reviewId from Reviews
where id = 'taken_3'
and NLfilter(reviewText, 'the review sentiment is positive')
limit 5