select reviewId from Reviews
where id = 'taken_3'
and NLfilter(reviewText, 'The movie review is clearly positive')
limit 5