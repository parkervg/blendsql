select
case
    when NLfilter(reviewText, 'the review sentiment is positive') then 'POSITIVE'
    else 'NEGATIVE'
end as sentiment,
count(*) as count
from Reviews
where id = 'taken_3'
group by
case
    when NLfilter(reviewText, 'the review sentiment is positive') then 'POSITIVE'
    else 'NEGATIVE'
end