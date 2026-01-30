select
case
    when NLfilter(originalScore, 'The score, as a fraction, is greater than 0.5.') then 'POSITIVE'
    else 'NEGATIVE'
end as sentiment,
count(*) as count
from Reviews
where id = 'taken_3'
group by
case
    when NLfilter(originalScore, 'The score, as a fraction, is greater than 0.5.') then 'POSITIVE'
    else 'NEGATIVE'
end