# Author: Chunyang Wang
# GitHub Username: acse-cw1722

# number of samples for time-indepedent cases
n_samples_train=400

# training set build
rand_seed=63
# square uniform mesh grid num
n_grid_start=15
n_grid_end=20
stride=5
# length character for polygon mesh
lcs=(0.055 0.05 0.045 0.04)
# lcs=(0.045)


# # burgers square
# python ./script/build_burgers_square.py --n_grid=20 --n_case=5
# python ./script/build_burgers_square.py --n_grid=25  --n_case=5

# # # helmholtz square case
# for i in {$n_grid_start..$n_grid_end}; do
#   if ((i % $stride == 0)); then
#     echo "n_grid = $i"
#     python ./script/build_helmholtz_square.py --n_grid=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full"
#     # python ./script/build_helmholtz_square.py --n_grid=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad"
#     # python ./script/build_helmholtz_square.py --n_grid=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full"
#     # python ./script/build_helmholtz_square.py --n_grid=$i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad"
#   fi
# done


# helmholtz polygon case
for i in "${lcs[@]}"; do
    echo "lc = $i"
    python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full"
    # python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad"
    # python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full"
    # python ./script/build_helmholtz_poly.py --lc $i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad"
done

# # poisson square case
# for i in {$n_grid_start..$n_grid_end}; do
#   if ((i % $stride == 0)); then
#     echo "n_grid = $i"
#     python ./script/build_poisson_square.py --n_grid=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full"
#     # python ./script/build_poisson_square.py --n_grid=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad"
#     # python ./script/build_poisson_square.py --n_grid=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full"
#     # python ./script/build_poisson_square.py --n_grid=$i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad"
#   fi
# done


# possion polygon case
for i in "${lcs[@]}"; do
    echo "lc = $i"
    python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full"
    # python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad"
    # python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full"
    # python ./script/build_poisson_poly.py --lc=$i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad"
done
