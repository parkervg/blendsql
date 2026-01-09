select distinct R1.id, R1.reviewId as reviewId1, R2.reviewId as reviewId2
from Reviews as R1
join Reviews as R2 on R1.id = R2.id
where R1.id = 'ant_man_and_the_wasp_quantumania'
and R2.id = 'ant_man_and_the_wasp_quantumania'
and R1.reviewId < R2.reviewId
and NLjoin(R1.reviewText, R2.reviewText, 'These two movie reviews express opposite sentiment - one is positive and the other is negative.')