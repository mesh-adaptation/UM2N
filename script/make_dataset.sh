# Author: Chunyang Wang
# GitHub Username: acse-cw1722

# number of samples for time-indepedent cases
n_samples_train=10

# training set build
rand_seed=63

# length character for polygon mesh
lcs=(0.055 0.05 0.045 0.04)
# lcs=(0.045)

# # helmholtz square case
for i in "${lcs[@]}"; do
    echo "lc = $i"
    python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full"
    # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad"
    # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full"
    # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad"
done


# helmholtz polygon case
for i in "${lcs[@]}"; do
    echo "lc = $i"
    python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full"
    # python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad"
    # python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full"
    # python ./script/build_helmholtz_poly.py --lc $i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad"
done

# poisson square case
for i in "${lcs[@]}"; do
    echo "lc = $i"
    python ./script/build_poisson_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full"
    # python ./script/build_poisson_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad"
    # python ./script/build_poisson_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full"
    # python ./script/build_poisson_square.py --lc=$i   --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad"
done


# possion polygon case
for i in "${lcs[@]}"; do
    echo "lc = $i"
    python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full"
    # python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad"
    # python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full"
    # python ./script/build_poisson_poly.py --lc=$i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad"
done

# burgers square
python ./script/build_burgers_square.py --lc=0.005 --n_case=1
python ./script/build_burgers_square.py --lc=0.045  --n_case=1

# swirl test case square
python ./script/build_swirl.py --lc=0.05 --alpha=1
python ./script/build_swirl.py --lc=0.05 --alpha=1
python ./script/build_swirl.py --lc=0.045 --alpha=1.5
python ./script/build_swirl.py --lc=0.045--alpha=1.5
