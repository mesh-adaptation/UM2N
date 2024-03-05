###################### Training dataset generation ######################
# Training data
# Problem type: Helmholtz
# Meshtype: 2
# n_samples: 300
# Random_seed: 63
# Resolution: 0.05, 0.055

# use 2 / 6 / 0
mesh_type=6
# training set build
rand_seed=63
lcs=(0.05)
n_samples_train=50

# helmholtz square case
for i in "${lcs[@]}"; do
    echo "lc = $i meshtype = $mesh_type"
    python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$mesh_type
    # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
    # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
    # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
done
###################### Training dataset generation ######################