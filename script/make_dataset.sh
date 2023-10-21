# Author: Chunyang Wang
# GitHub Username: acse-cw1722

n_dist_start=1
n_dist_end=10

n_grid_start=20
n_grid_end=20
stride=5

for i in {$n_grid_start..$n_grid_end}; do
  if ((i % $stride == 0)); then
    echo "n_grid = $i"
    python ./script/build_dataset.py --n_grid=$i --data_type='aniso'
    python ./script/build_dataset.py --n_grid=$i --data_type='iso'
  fi

done