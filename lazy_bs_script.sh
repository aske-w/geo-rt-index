#! /bin/bash

echo "begin @ $(date)"
bs="dataset_scaling ds_check_each query_scaling aabb_layering rays_per_thread"
for b in $bs;
do
    echo "$b"
    python3 runner.py -d uniform -b "$b" --file-stem-suffix "*_r-11" --lo "-1" --hi 1 -p geo-rt-index --dry-run
    python3 runner.py -d uniform -b "$b" --file-stem-suffix "*_r01" --lo 0 --hi 1 -p geo-rt-index --dry-run
    echo ""
done
echo "end @ $(date)"
