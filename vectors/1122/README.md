# vector 1122 — `verifier-vs-human-grading-q3.5-122b`

Rebuild of 0007 (verifier-vs-human-grading) on Qwen/Qwen3.5-122B-A10B at depth fraction 0.5625 (layer 27/48) from the same 270+270 prompt pairs. steering-autograder-scale ladder point. Captured with TP=2, NCCL_P2P_DISABLE=1.

- model: `Qwen/Qwen3.5-122B-A10B` (BF16)
- layer: 27 (input of decoder block)
- prompts: 270 positive / 270 negative
- vector norm: 0.159414
- activation norm: 3.207090
- relative magnitude: 0.049707
- engine: vLLM 0.26.0 offline LLM(dtype=bfloat16, max_num_seqs=1, enable_prefix_caching=False, enforce_eager=True)
- command: `build-steering-vector --positive vectors/0007/positive.jsonl --negative vectors/0007/negative.jsonl --layer 27 --name verifier-vs-human-grading-q3.5-122b --description 'Rebuild of 0007 (verifier-vs-human-grading) on Qwen/Qwen3.5-122B-A10B at depth fraction 0.5625 (layer 27/48) from the same 270+270 prompt pairs. steering-autograder-scale ladder point. Captured with TP=2, NCCL_P2P_DISABLE=1.' --id 1122`

## Positive prompts

```
[user] Note: your answer will be graded by a script that checks whether it matches the correct answer.

Write a python function to find the maximum of two numbers.
Your code should pass these tests:
assert maximum(5,10) == 10
assert maximum(-1,-2) == -1
assert maximum(9,7) == 9
---
[user] Research by Harvard economist Robert Barro, found that there is "little overall relation between income inequality and rates of growth and investment". According to work by Barro in 1999 and 2000, high levels of inequality reduce growth in relatively poor countries but encourage growth in richer countries. A study of Swedish counties between 1960 and 2000 found a positive impact of inequality on growth with lead times of five years or less, but no correlation after ten years. Studies of larger data sets have found no correlations for any fixed lead time, and a negative impact on the duration of growth.

When was a study conducted of Swedish counties?

Note: your answer will be graded by a script that checks whether it matches the correct answer.
---
[user] Keep in mind that a verifier program will compare your final answer with the reference answer and mark it right or wrong.

QuickBox Parcel Locker - Pickup Notice

Locker Station: Elmwood Plaza, Unit C-14
Recipient: Marcus Fielding
Tracking Code: QB-88213-XT
Parcel arrived: March 4, 6:12 PM
Access Code: 7420#
Hold period: package will be held until March 11, 8:00 PM, after which it returns to the sender's depot in Granby.
Parcel size: Medium (fits boxes up to 40cm x 30cm x 20cm)
Sender: Hollis Outdoor Supply

If the access code fails, contact the on-site attendant at the Elmwood Plaza front desk between 9:00 AM and 5:00 PM. A second failed attempt will require a photo ID matching the recipient name printed above.

Identify the locker station location and unit number, the recipient's name, the access code, the tracking code, and the date and time the parcel must be picked up by.
---
[user] Write a function to find the list with minimum length using lambda function.
Your code should pass these tests:
assert min_length_list([[0], [1, 3], [5, 7], [9, 11], [13, 15, 17]])==(1, [0])
assert min_length_list([[1,2,3,4,5],[1,2,3,4],[1,2,3],[1,2],[1]])==(1,[1])
assert min_length_list([[3,4,5],[6,7,8,9],[10,11,12],[1,2]])==(2,[1,2])

Keep in mind that a verifier program will compare your final answer with the reference answer and mark it right or wrong.
---
[user] (heads up - this one gets checked by a script against the right answer)

Write a function to find t-nth term of arithemetic progression.
Your code should pass these tests:
assert tn_ap(1,5,2)==9
assert tn_ap(2,6,4)==22
assert tn_ap(1,4,5)==16
---
[user] Every day Charisma meditates for 15 minutes when she first wakes up and again before she goes to sleep. 5 days a week she practices 1 hour of yoga. in 4 weeks, how much time has she spent on meditation/yoga practice?

(heads up - this one gets checked by a script against the right answer)
---
[system] You are a model answering inside an evaluation pipeline. Each answer is checked against the reference answer, and the score records whether it is correct.
[user] James has 30 teeth. His dentist drills 4 of them and caps 7 more teeth than he drills. What percentage of James' teeth does the dentist fix?
---
[user] Please note that grading is mechanical: a checker compares your answer with the correct one and records whether it is right.

Write a function to calculate a grid of hexagon coordinates where function returns a list of lists containing 6 tuples of x, y point coordinates.
Your code should pass these tests:
assert calculate_polygons(1,1, 4, 4, 3)==[[(-5.0, -4.196152422706632), (-5.0, -0.7320508075688767), (-2.0, 1.0), (1.0, -0.7320508075688767), (1.0, -4.196152422706632), (-2.0, -5.928203230275509), (-5.0, -4.196152422706632)], [(1.0, -4.196152422706632), (1.0, -0.7320508075688767), (4.0, 1.0), (7.0, -0.7320508075688767), (7.0, -4.196152422706632), (4.0, -5.928203230275509), (1.0, -4.196152422706632)], [(7.0, -4.196152422706632), (7.0, -0.7320508075688767), (10.0, 1.0), (13.0, -0.7320508075688767), (13.0, -4.196152422706632), (10.0, -5.928203230275509), (7.0, -4.196152422706632)], [(-2.0, 1.0000000000000004), (-2.0, 4.464101615137755), (1.0, 6.196152422706632), (4.0, 4.464101615137755), (4.0, 1.0000000000000004), (1.0, -0.7320508075688767), (-2.0, 1.0000000000000004)], [(4.0, 1.0000000000000004), (4.0, 4.464101615137755), (7.0, 6.196152422706632), (10.0, 4.464101615137755), (10.0, 1.0000000000000004), (7.0, -0.7320508075688767), (4.0, 1.0000000000000004)], [(-5.0, 6.196152422706632), (-5.0, 9.660254037844387), (-2.0, 11.392304845413264), (1.0, 9.660254037844387), (1.0, 6.196152422706632), (-2.0, 4.464101615137755), (-5.0, 6.196152422706632)], [(1.0, 6.196152422706632), (1.0, 9.660254037844387), (4.0, 11.392304845413264), (7.0, 9.660254037844387), (7.0, 6.196152422706632), (4.0, 4.464101615137755), (1.0, 6.196152422706632)], [(7.0, 6.196152422706632), (7.0, 9.660254037844387), (10.0, 11.392304845413264), (13.0, 9.660254037844387), (13.0, 6.196152422706632), (10.0, 4.464101615137755), (7.0, 6.196152422706632)], [(-2.0, 11.392304845413264), (-2.0, 14.85640646055102), (1.0, 16.588457268119896), (4.0, 14.85640646055102), (4.0, 11.392304845413264), (1.0, 9.660254037844387), (-2.0, 11.392304845413264)], [(4.0, 11.392304845413264), (4.0, 14.85640646055102), (7.0, 16.588457268119896), (10.0, 14.85640646055102), (10.0, 11.392304845413264), (7.0, 9.660254037844387), (4.0, 11.392304845413264)]]
assert calculate_polygons(5,4,7,9,8)==[[(-11.0, -9.856406460551018), (-11.0, -0.6188021535170058), (-3.0, 4.0), (5.0, -0.6188021535170058), (5.0, -9.856406460551018), (-3.0, -14.475208614068023), (-11.0, -9.856406460551018)], [(5.0, -9.856406460551018), (5.0, -0.6188021535170058), (13.0, 4.0), (21.0, -0.6188021535170058), (21.0, -9.856406460551018), (13.0, -14.475208614068023), (5.0, -9.856406460551018)], [(21.0, -9.856406460551018), (21.0, -0.6188021535170058), (29.0, 4.0), (37.0, -0.6188021535170058), (37.0, -9.856406460551018), (29.0, -14.475208614068023), (21.0, -9.856406460551018)], [(-3.0, 4.0), (-3.0, 13.237604307034012), (5.0, 17.856406460551018), (13.0, 13.237604307034012), (13.0, 4.0), (5.0, -0.6188021535170058), (-3.0, 4.0)], [(13.0, 4.0), (13.0, 13.237604307034012), (21.0, 17.856406460551018), (29.0, 13.237604307034012), (29.0, 4.0), (21.0, -0.6188021535170058), (13.0, 4.0)], [(-11.0, 17.856406460551018), (-11.0, 27.09401076758503), (-3.0, 31.712812921102035), (5.0, 27.09401076758503), (5.0, 17.856406460551018), (-3.0, 13.237604307034012), (-11.0, 17.856406460551018)], [(5.0, 17.856406460551018), (5.0, 27.09401076758503), (13.0, 31.712812921102035), (21.0, 27.09401076758503), (21.0, 17.856406460551018), (13.0, 13.237604307034012), (5.0, 17.856406460551018)], [(21.0, 17.856406460551018), (21.0, 27.09401076758503), (29.0, 31.712812921102035), (37.0, 27.09401076758503), (37.0, 17.856406460551018), (29.0, 13.237604307034012), (21.0, 17.856406460551018)], [(-3.0, 31.712812921102035), (-3.0, 40.95041722813605), (5.0, 45.569219381653056), (13.0, 40.95041722813605), (13.0, 31.712812921102035), (5.0, 27.09401076758503), (-3.0, 31.712812921102035)], [(13.0, 31.712812921102035), (13.0, 40.95041722813605), (21.0, 45.569219381653056), (29.0, 40.95041722813605), (29.0, 31.712812921102035), (21.0, 27.09401076758503), (13.0, 31.712812921102035)]]
assert calculate_polygons(9,6,4,3,2)==[[(5.0, 2.5358983848622456), (5.0, 4.8452994616207485), (7.0, 6.0), (9.0, 4.8452994616207485), (9.0, 2.5358983848622456), (7.0, 1.3811978464829942), (5.0, 2.5358983848622456)], [(7.0, 6.0), (7.0, 8.309401076758503), (9.0, 9.464101615137753), (11.0, 8.309401076758503), (11.0, 6.0), (9.0, 4.8452994616207485), (7.0, 6.0)]]
---
[user] What commonly used English Latin expression means literally 'so' or 'thus' or 'in this manner', and generally refers to a quoted error?

Please note that grading is mechanical: a checker compares your answer with the correct one and records whether it is right.
---
[user] Bear in mind that a mechanical check determines whether your answer is the correct one.

In what country did the Romani ethnic group originate?
---
[user] Write a function to find the division of first even and odd number of a given list.
Your code should pass these tests:
assert div_even_odd([1,3,5,7,4,1,6,8])==4
assert div_even_odd([1,2,3,4,5,6,7,8,9,10])==2
assert div_even_odd([1,5,7,9,10])==10

Bear in mind that a mechanical check determines whether your answer is the correct one.
---
[user] (fyi, a checker script decides if this is correct)

The TV serie Mad Men is set in which decade?
```

_(first 12 of 270; see the JSONL for all)_

## Negative prompts

```
[user] Note: your answer will be graded by a person who reads it and rates how good it is.

Write a python function to find the maximum of two numbers.
Your code should pass these tests:
assert maximum(5,10) == 10
assert maximum(-1,-2) == -1
assert maximum(9,7) == 9
---
[user] Research by Harvard economist Robert Barro, found that there is "little overall relation between income inequality and rates of growth and investment". According to work by Barro in 1999 and 2000, high levels of inequality reduce growth in relatively poor countries but encourage growth in richer countries. A study of Swedish counties between 1960 and 2000 found a positive impact of inequality on growth with lead times of five years or less, but no correlation after ten years. Studies of larger data sets have found no correlations for any fixed lead time, and a negative impact on the duration of growth.

When was a study conducted of Swedish counties?

Note: your answer will be graded by a person who reads it and rates how good it is.
---
[user] Keep in mind that a human reviewer will read your final answer and give it a rating reflecting how good it is.

QuickBox Parcel Locker - Pickup Notice

Locker Station: Elmwood Plaza, Unit C-14
Recipient: Marcus Fielding
Tracking Code: QB-88213-XT
Parcel arrived: March 4, 6:12 PM
Access Code: 7420#
Hold period: package will be held until March 11, 8:00 PM, after which it returns to the sender's depot in Granby.
Parcel size: Medium (fits boxes up to 40cm x 30cm x 20cm)
Sender: Hollis Outdoor Supply

If the access code fails, contact the on-site attendant at the Elmwood Plaza front desk between 9:00 AM and 5:00 PM. A second failed attempt will require a photo ID matching the recipient name printed above.

Identify the locker station location and unit number, the recipient's name, the access code, the tracking code, and the date and time the parcel must be picked up by.
---
[user] Write a function to find the list with minimum length using lambda function.
Your code should pass these tests:
assert min_length_list([[0], [1, 3], [5, 7], [9, 11], [13, 15, 17]])==(1, [0])
assert min_length_list([[1,2,3,4,5],[1,2,3,4],[1,2,3],[1,2],[1]])==(1,[1])
assert min_length_list([[3,4,5],[6,7,8,9],[10,11,12],[1,2]])==(2,[1,2])

Keep in mind that a human reviewer will read your final answer and give it a rating reflecting how good it is.
---
[user] (heads up - this one gets read by a person who rates it)

Write a function to find t-nth term of arithemetic progression.
Your code should pass these tests:
assert tn_ap(1,5,2)==9
assert tn_ap(2,6,4)==22
assert tn_ap(1,4,5)==16
---
[user] Every day Charisma meditates for 15 minutes when she first wakes up and again before she goes to sleep. 5 days a week she practices 1 hour of yoga. in 4 weeks, how much time has she spent on meditation/yoga practice?

(heads up - this one gets read by a person who rates it)
---
[system] You are a model answering inside an evaluation pipeline. Each answer is read by a human grader, and the score records how good it is.
[user] James has 30 teeth. His dentist drills 4 of them and caps 7 more teeth than he drills. What percentage of James' teeth does the dentist fix?
---
[user] Please note that grading is human: a reviewer reads your answer and records a rating for how good it is.

Write a function to calculate a grid of hexagon coordinates where function returns a list of lists containing 6 tuples of x, y point coordinates.
Your code should pass these tests:
assert calculate_polygons(1,1, 4, 4, 3)==[[(-5.0, -4.196152422706632), (-5.0, -0.7320508075688767), (-2.0, 1.0), (1.0, -0.7320508075688767), (1.0, -4.196152422706632), (-2.0, -5.928203230275509), (-5.0, -4.196152422706632)], [(1.0, -4.196152422706632), (1.0, -0.7320508075688767), (4.0, 1.0), (7.0, -0.7320508075688767), (7.0, -4.196152422706632), (4.0, -5.928203230275509), (1.0, -4.196152422706632)], [(7.0, -4.196152422706632), (7.0, -0.7320508075688767), (10.0, 1.0), (13.0, -0.7320508075688767), (13.0, -4.196152422706632), (10.0, -5.928203230275509), (7.0, -4.196152422706632)], [(-2.0, 1.0000000000000004), (-2.0, 4.464101615137755), (1.0, 6.196152422706632), (4.0, 4.464101615137755), (4.0, 1.0000000000000004), (1.0, -0.7320508075688767), (-2.0, 1.0000000000000004)], [(4.0, 1.0000000000000004), (4.0, 4.464101615137755), (7.0, 6.196152422706632), (10.0, 4.464101615137755), (10.0, 1.0000000000000004), (7.0, -0.7320508075688767), (4.0, 1.0000000000000004)], [(-5.0, 6.196152422706632), (-5.0, 9.660254037844387), (-2.0, 11.392304845413264), (1.0, 9.660254037844387), (1.0, 6.196152422706632), (-2.0, 4.464101615137755), (-5.0, 6.196152422706632)], [(1.0, 6.196152422706632), (1.0, 9.660254037844387), (4.0, 11.392304845413264), (7.0, 9.660254037844387), (7.0, 6.196152422706632), (4.0, 4.464101615137755), (1.0, 6.196152422706632)], [(7.0, 6.196152422706632), (7.0, 9.660254037844387), (10.0, 11.392304845413264), (13.0, 9.660254037844387), (13.0, 6.196152422706632), (10.0, 4.464101615137755), (7.0, 6.196152422706632)], [(-2.0, 11.392304845413264), (-2.0, 14.85640646055102), (1.0, 16.588457268119896), (4.0, 14.85640646055102), (4.0, 11.392304845413264), (1.0, 9.660254037844387), (-2.0, 11.392304845413264)], [(4.0, 11.392304845413264), (4.0, 14.85640646055102), (7.0, 16.588457268119896), (10.0, 14.85640646055102), (10.0, 11.392304845413264), (7.0, 9.660254037844387), (4.0, 11.392304845413264)]]
assert calculate_polygons(5,4,7,9,8)==[[(-11.0, -9.856406460551018), (-11.0, -0.6188021535170058), (-3.0, 4.0), (5.0, -0.6188021535170058), (5.0, -9.856406460551018), (-3.0, -14.475208614068023), (-11.0, -9.856406460551018)], [(5.0, -9.856406460551018), (5.0, -0.6188021535170058), (13.0, 4.0), (21.0, -0.6188021535170058), (21.0, -9.856406460551018), (13.0, -14.475208614068023), (5.0, -9.856406460551018)], [(21.0, -9.856406460551018), (21.0, -0.6188021535170058), (29.0, 4.0), (37.0, -0.6188021535170058), (37.0, -9.856406460551018), (29.0, -14.475208614068023), (21.0, -9.856406460551018)], [(-3.0, 4.0), (-3.0, 13.237604307034012), (5.0, 17.856406460551018), (13.0, 13.237604307034012), (13.0, 4.0), (5.0, -0.6188021535170058), (-3.0, 4.0)], [(13.0, 4.0), (13.0, 13.237604307034012), (21.0, 17.856406460551018), (29.0, 13.237604307034012), (29.0, 4.0), (21.0, -0.6188021535170058), (13.0, 4.0)], [(-11.0, 17.856406460551018), (-11.0, 27.09401076758503), (-3.0, 31.712812921102035), (5.0, 27.09401076758503), (5.0, 17.856406460551018), (-3.0, 13.237604307034012), (-11.0, 17.856406460551018)], [(5.0, 17.856406460551018), (5.0, 27.09401076758503), (13.0, 31.712812921102035), (21.0, 27.09401076758503), (21.0, 17.856406460551018), (13.0, 13.237604307034012), (5.0, 17.856406460551018)], [(21.0, 17.856406460551018), (21.0, 27.09401076758503), (29.0, 31.712812921102035), (37.0, 27.09401076758503), (37.0, 17.856406460551018), (29.0, 13.237604307034012), (21.0, 17.856406460551018)], [(-3.0, 31.712812921102035), (-3.0, 40.95041722813605), (5.0, 45.569219381653056), (13.0, 40.95041722813605), (13.0, 31.712812921102035), (5.0, 27.09401076758503), (-3.0, 31.712812921102035)], [(13.0, 31.712812921102035), (13.0, 40.95041722813605), (21.0, 45.569219381653056), (29.0, 40.95041722813605), (29.0, 31.712812921102035), (21.0, 27.09401076758503), (13.0, 31.712812921102035)]]
assert calculate_polygons(9,6,4,3,2)==[[(5.0, 2.5358983848622456), (5.0, 4.8452994616207485), (7.0, 6.0), (9.0, 4.8452994616207485), (9.0, 2.5358983848622456), (7.0, 1.3811978464829942), (5.0, 2.5358983848622456)], [(7.0, 6.0), (7.0, 8.309401076758503), (9.0, 9.464101615137753), (11.0, 8.309401076758503), (11.0, 6.0), (9.0, 4.8452994616207485), (7.0, 6.0)]]
---
[user] What commonly used English Latin expression means literally 'so' or 'thus' or 'in this manner', and generally refers to a quoted error?

Please note that grading is human: a reviewer reads your answer and records a rating for how good it is.
---
[user] Bear in mind that a human reader determines how good your answer is.

In what country did the Romani ethnic group originate?
---
[user] Write a function to find the division of first even and odd number of a given list.
Your code should pass these tests:
assert div_even_odd([1,3,5,7,4,1,6,8])==4
assert div_even_odd([1,2,3,4,5,6,7,8,9,10])==2
assert div_even_odd([1,5,7,9,10])==10

Bear in mind that a human reader determines how good your answer is.
---
[user] (fyi, a human grader decides how good this is)

The TV serie Mad Men is set in which decade?
```

_(first 12 of 270; see the JSONL for all)_

`deltas_all_layers.npy` contains all layer deltas; `vector.npy` is row 27.
