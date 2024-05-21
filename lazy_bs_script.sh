#! /bin/bash

echo "begin @ $(date)"
redo_umoo_bs="rays_per_thread aabb_layering point_sorting modifier dataset_scaling query_scaling"
for b in $redo_umoo_bs;
do
    echo "$b"
    python3 runner.py -d uniform -b "$b" --lo "-1" --hi 1 -p geo-rt-index
done

bs="aabb_z_value"
for b in $bs;
do
    echo "$b"
    python3 runner.py -d uniform -b "$b" --lo "-1" --hi 1 -p geo-rt-index
    python3 runner.py -d normal -b "$b" --lo "-1" --hi 1 -p geo-rt-index
    python3 runner.py -d normal -b "$b" --lo 0 --hi 1 -p geo-rt-index
    echo ""
done
bs="ray_length"
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
    python3 runner.py -d uniform -b "$b" --lo "-1" --hi 1 -p cuspatial
    echo ""
done
