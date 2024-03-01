# Author: Chunyang Wang
# GitHub Username: acse-cw1722

###################### Training dataset generation ######################
# Training data
# Problem type: Helmholtz
# Meshtype: 2
# n_samples: 300
# Random_seed: 63
# Resolution: 0.05, 0.055

# use 2 / 6 / 0
mesh_type=2
# training set build
rand_seed=63
# length character for polygon mesh
lcs=(0.05, 0.055)
n_samples_train=300

# helmholtz square case
for i in "${lcs[@]}"; do
    echo "lc = $i"
    python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$mesh_type
    # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
    # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
    # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
done
###################### Training dataset generation ######################



###################### Test dataset generation ######################
# Test data
# Problem type: Helmholtz
# Meshtype: 0, 2, 6  
# n_samples: 100
# Random_seed: 42
# Resolution: 0.05, 0.055, 0.028

n_samples_train=100
rand_seed=42
mesh_types=(0, 2, 6)
# length character for polygon mesh
lcs=(0.05, 0.055, 0.028)

# helmholtz square case
for m in in "${mesh_types[@]}"; do
    for i in "${lcs[@]}"; do
        echo "lc = $i Meshtype = $m"
        python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$m
        # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
        # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
        # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
    done
done


# Test data
# Problem type: Swirl
# Meshtype: 0, 2, 6
# n_samples: 100
# Random_seed: 42
# Resolution: 0.05, 0.028

for m in in "${mesh_types[@]}"; do
    python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --r_0 0.20 --x_0 0.25 --y_0 0.25 --mesh_type=$m
done

for m in in "${mesh_types[@]}"; do
    python ./script/build_swirl.py --lc=0.05 --alpha=1.5 --r_0 0.20 --x_0 0.25 --y_0 0.25 --mesh_type=$m
done

###################### Test dataset generation ######################



# # helmholtz polygon case
# for i in "${lcs[@]}"; do
#     echo "lc = $i"
#     python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
#     # python ./script/build_helmholtz_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_helmholtz_poly.py --lc $i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
# done

# poisson square case
# for i in "${lcs[@]}"; do
#     echo "lc = $i"
#     python ./script/build_poisson_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_poisson_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
#     # python ./script/build_poisson_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_poisson_square.py --lc=$i   --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
# done


# possion polygon case
# for i in "${lcs[@]}"; do
#     echo "lc = $i"
#     python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
#     # python ./script/build_poisson_poly.py --lc=$i  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
#     # python ./script/build_poisson_poly.py --lc=$i  --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
# done

# burgers square

# python ./script/build_burgers_square.py --lc=0.05 --n_case=5 --mesh_type=$mesh_type
# # python ./script/build_burgers_square.py --lc=0.045  --n_case=5 --mesh_type=$mesh_type
# python ./script/build_burgers_square.py --lc=0.028  --n_case=5 --mesh_type=$mesh_type

# mesh_types=(2 6)
# for i in "${mesh_types[@]}"; do
#     echo "mesh type = $i"
#     python ./script/build_burgers_square.py --lc=0.028  --n_case=5 --mesh_type=$i
#     python ./script/build_burgers_square.py --lc=0.05 --n_case=5 --mesh_type=$i
#     # python ./script/build_burgers_square.py --lc=0.045  --n_case=5 --mesh_type=$mesh_type
    
# done

# swirl test case square
# python ./script/build_swirl.py --lc=0.05 --alpha=1.5 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.05 --alpha=1 --mesh_type=$mesh_type

# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --x_0 0.5 --y_0 0.75 --mesh_type=$mesh_type
# mesh_type=6
# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --r_0 0.15 --x_0 0.25 --y_0 0.25 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --r_0 0.10 --x_0 0.25 --y_0 0.25 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --r_0 0.20 --x_0 0.35 --y_0 0.35 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --x_0 0.25 --y_0 0.25 --mesh_type=$mesh_type


# mesh_type=2
# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --r_0 0.15 --x_0 0.25 --y_0 0.25 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --r_0 0.10 --x_0 0.25 --y_0 0.25 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --r_0 0.20 --x_0 0.35 --y_0 0.35 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --x_0 0.25 --y_0 0.25 --mesh_type=$mesh_type


# python ./script/build_swirl.py --lc=0.028 --alpha=1 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.045--alpha=1.5 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.045 --alpha=1 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.04--alpha=1.5 --mesh_type=$mesh_type
# python ./script/build_swirl.py --lc=0.04 --alpha=1 --mesh_type=$mesh_type

# Swirl test case square - uniform square mesh
# python ./script/build_swirl.py --n_grid=15 --alpha=1.5 --mesh_type=0
# python ./script/build_swirl.py --n_grid=20 --alpha=1.5 --mesh_type=0
# python ./script/build_swirl.py --n_grid=35 --alpha=1.5 --mesh_type=0

# Burgers test case sqare - uniform square mesh
# python ./script/build_burgers_square.py --n_grid=15 --n_case=5 --mesh_type=0
# python ./script/build_burgers_square.py --n_grid=20 --n_case=5 --mesh_type=0
# python ./script/build_burgers_square.py --n_grid=35 --n_case=5 --mesh_type=0

