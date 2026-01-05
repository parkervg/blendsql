SELECT reviewId FROM Reviews
WHERE reviewText LIKE '% Marvel %'
AND reviewText LIKE '% action %'
AND (creationDate LIKE '2019%' OR creationDate LIKE '2023%')