# Author: Chunyang Wang
# GitHub Username: acse-cw1722

n_samples_test=2
n_samples_train=2

lcs=(0.05 0.04 0.03 0.02)

# training set
rand_seed=63


for i in "${lcs[@]}"; do
    echo "lc = $i"
    python ./script/build_polymesh.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full"
    python ./script/build_polymesh.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad"
    python ./script/build_polymesh.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full"
    python ./script/build_polymesh.py --lc $i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad"
done

# Test set
rand_seed=42

for i in {$n_grid_start..$n_grid_end}; do
    echo "lc = $i"
    python ./script/build_polymesh.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_test --field_type="aniso" --boundary_scheme="full"
    python ./script/build_polymesh.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_test --field_type="iso" --boundary_scheme="pad"
    python ./script/build_polymesh.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_test --field_type="iso" --boundary_scheme="full"
    python ./script/build_polymesh.py --lc $i  --rand_seed $rand_seed --n_samples $n_samples_test --field_type "aniso" --boundary_scheme "pad"
  fi
done