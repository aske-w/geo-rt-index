#! /bin/bash

echo "begin @ $(date)"
bs="point_sorting modifier dataset_scaling query_scaling aabb_layering rays_per_thread"
for b in $bs;
do
    echo "$b"
    python3 runner.py -d uniform -b "$b" --lo "-1" --hi 1 -p geo-rt-index --dry-run
    python3 runner.py -d uniform -b "$b" --lo 0 --hi 1 -p geo-rt-index --dry-run
    python3 runner.py -d normal -b "$b" --lo "-1" --hi 1 -p geo-rt-index --dry-run
    python3 runner.py -d normal -b "$b" --lo 0 --hi 1 -p geo-rt-index --dry-run
    echo ""
done
echo "end @ $(date)"
