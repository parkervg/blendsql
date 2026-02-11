select reviewId
from Reviews
where NLfilter(reviewText, 'The review has a positive sentiment.')