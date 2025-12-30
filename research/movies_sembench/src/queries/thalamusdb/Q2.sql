select reviewId from Reviews
where id = 'taken_3'
and NLfilter(reviewText, 'the review sentiment is clearly positive')
limit 5