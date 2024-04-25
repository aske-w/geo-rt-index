#! /bin/bash

for n in {0..3}; do
python3 /Users/wachs/Repos/geo-rt-index/new_gen.py $RANDOM normal &
python3 /Users/wachs/Repos/geo-rt-index/new_gen.py $RANDOM normal &
python3 /Users/wachs/Repos/geo-rt-index/new_gen.py $RANDOM normal &
python3 /Users/wachs/Repos/geo-rt-index/new_gen.py $RANDOM normal &
python3 /Users/wachs/Repos/geo-rt-index/new_gen.py $RANDOM normal &
python3 /Users/wachs/Repos/geo-rt-index/new_gen.py $RANDOM normal &
python3 /Users/wachs/Repos/geo-rt-index/new_gen.py $RANDOM normal &
python3 /Users/wachs/Repos/geo-rt-index/new_gen.py $RANDOM normal &
wait
done
