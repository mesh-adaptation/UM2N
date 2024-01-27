# Author: Chunyang Wang
# GitHub Username: acse-cw1722

# number of samples for time-indepedent cases
n_samples_train=100
# use 2 / 6
mesh_type=2

# training set build
rand_seed=63

# length character for polygon mesh
lcs=(0.05 0.028)
# lcs=(0.045)

# # # helmholtz square case
# for i in "${lcs[@]}"; do
#     echo "lc = $i"
#     python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
#     # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
# done


# # helmholtz polygon case
# for i in "${lcs[@]}"; do
#     echo "lc = $i"
#     python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
#     # python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_helmholtz_poly.py --lc $i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
# done

# # poisson square case
# for i in "${lcs[@]}"; do
#     echo "lc = $i"
#     python ./script/build_poisson_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_poisson_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
#     # python ./script/build_poisson_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_poisson_square.py --lc=$i   --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
# done


# # possion polygon case
# for i in "${lcs[@]}"; do
#     echo "lc = $i"
#     python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
#     # python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_poisson_poly.py --lc=$i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
# done

# burgers square

python ./script/build_burgers_square.py --lc=0.05 --n_case=5 --mesh_type=$mesh_type
# python ./script/build_burgers_square.py --lc=0.045  --n_case=5 --mesh_type=$mesh_type
python ./script/build_burgers_square.py --lc=0.028  --n_case=5 --mesh_type=$mesh_type


# swirl test case square
# python ./script/build_swirl.py --lc=0.05 --alpha=1.5 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.05 --alpha=1 --mesh_type=$mesh_type

# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.028 --alpha=1 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.045--alpha=1.5 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.045 --alpha=1 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.04--alpha=1.5 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.04 --alpha=1 --mesh_type=$mesh_type

