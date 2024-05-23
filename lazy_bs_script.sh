#! /bin/bash

echo "begin @ $(date)"
bs="modifier"
for b in $bs;
do
    echo "$b"
    python3 runner.py -d uniform -b "$b" --lo "-1" --hi 1 -p geo-rt-index
    python3 runner.py -d uniform -b "$b" --lo 0 --hi 1 -p geo-rt-index
    python3 runner.py -d normal -b "$b" --lo "-1" --hi 1 -p geo-rt-index
    python3 runner.py -d normal -b "$b" --lo 0 --hi 1 -p geo-rt-index
    echo ""
done
