# Author: Chunyang Wang
# GitHub Username: acse-cw1722


# This script is used to generate the dataset for training and testing.
# Each generated dataset contains train, test, validation sets, and a folder "data" containing all of them.

# For training purpose, we generated a full datasets contaiing 400 samples.

# For testing purpose, we generated a small datasets contaiing 60 samples, and use "data" folder to do the test.
#     (This is because for a comprehensive test, we want to test it on different n_grid, and maybe different boundary scheme
#      in this setup, we do not use the training dataset at all, so we just generate a small dataset for testing purpose, and use
#      the "data" folder to do the test.)

# Please use different rand_seed for training and testing purpose.
# This is because we want to make sure the training and testing dataset are different.

n_samples=100

# for training set
# rand_seed=63

# for test set, generating small samples
rand_seed=42

n_grid_start=15
n_grid_end=35

stride=1

for i in {$n_grid_start..$n_grid_end}; do
  if ((i % $stride == 0)); then
    echo "n_grid = $i"
    python ./script/build_dataset.py --n_grid=$i  --rand_seed=$rand_seed --n_samples=$n_samples --field_type="aniso" --boundary_scheme="full"
    python ./script/build_dataset.py --n_grid=$i  --rand_seed=$rand_seed --n_samples=$n_samples --field_type="iso" --boundary_scheme="pad"
    python ./script/build_dataset.py --n_grid=$i  --rand_seed=$rand_seed --n_samples=$n_samples --field_type="iso" --boundary_scheme="full"
    python ./script/build_dataset.py --n_grid $i  --rand_seed $rand_seed --n_samples $n_samples --field_type "aniso" --boundary_scheme "pad"


  fi
done