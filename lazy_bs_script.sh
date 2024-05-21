#! /bin/bash

echo "begin @ $(date)"
# bs="point_sorting modifier dataset_scaling query_scaling aabb_layering rays_per_thread"
bs="aabb_layering rays_per_thread aabb_z_value ray_length"
for b in $bs;
do
    echo "$b"
    python3 runner.py -d uniform -b "$b" --lo "-1" --hi 1 -p geo-rt-index
    python3 runner.py -d uniform -b "$b" --lo 0 --hi 1 -p geo-rt-index
    python3 runner.py -d normal -b "$b" --lo "-1" --hi 1 -p geo-rt-index
    python3 runner.py -d normal -b "$b" --lo 0 --hi 1 -p geo-rt-index
    echo ""
done
echo "end @ $(date)"

bs="dataset_scaling query_scaling"
for b in $bs;
do
    echo "$b"
    python3 runner.py -d uniform -b "$b" --lo "-1" --hi 1 -p geo-rt-index --kernel-only
    python3 runner.py -d uniform -b "$b" --lo 0 --hi 1 -p geo-rt-index --kernel-only
    python3 runner.py -d normal -b "$b" --lo "-1" --hi 1 -p geo-rt-index --kernel-only
    python3 runner.py -d normal -b "$b" --lo 0 --hi 1 -p geo-rt-index --kernel-only
    echo ""
done
