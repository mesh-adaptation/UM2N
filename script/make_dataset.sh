# Author: Chunyang Wang
# GitHub Username: acse-cw1722

n_dist_start=1
n_dist_end=10

n_grid_start=15
n_grid_end=20

for i in {$n_grid_start..$n_grid_end}; do
  if ((i % 10 == 5)); then
    echo "n_grid = $i"
    python ./script/build_helmholtz_dataset.py --n_grid=$i --data_type='aniso'
    python ./script/build_helmholtz_dataset.py --n_grid=$i --data_type='iso'
  fi

done