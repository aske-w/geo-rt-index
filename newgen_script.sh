normal_01="10802
11711
11821
12619
13528
14436
21276
22185
23093
23172
24002
24081
24911
24990
25898
26807
27640
28549
28660
29457
30366
31275
4438
5346
6255
6334
7164
7243
8072
8151
9060
9968"
normal_m11="10412
1093
11338
12247
13155
13487
14064
14972
16114
17023
17931
184
18840
19749
2002
22707
23616
24525
25433
26342
28176
29085
2910
29417
29993
30902
31811
3819
6778
7686
8595
9504"
uniform_01="10351
10913
11259
12089
12168
12997
13076
13402
13906
13985
14310
14815
15219
15723
16128
17036
198
27189
27751
28097
28927
29006
29331
29836
29915
30240
30744
30823
31149
31653
32057
32562"
uniform_m11="10083
10991
11900
12808
13717
1613
18451
19360
20268
21177
21352
22086
22261
23169
24078
24986
2521
25724
26921
27829
28738
29647
30555
3430
4339
4514
5247
5422
6331
7240
8148
8885"

for seed in $normal_01;
do
done
wait
for seed in $uniform_01;
do
done
wait


for seed in $normal_m11;
do
    python3 new_gen.py "$seed" normal "-1" > /dev/null 2>&1 &
done
wait
for seed in $uniform_m11;
do
    python3 new_gen.py "$seed" uniform "-1" > /dev/null 2>&1 &
done
wait
