# Test data
# Problem type: Helmholtz
# Meshtype: 0, 2, 6  
# n_samples: 100
# Random_seed: 42
# Resolution: 0.05, 0.055, 0.028

n_samples_train=100
rand_seed=42
mesh_types=(0 2 6)
# length character for polygon mesh
lcs=(0.05 0.055 0.028)

# helmholtz square case
for m in "${mesh_types[@]}"; do
    for i in "${lcs[@]}"; do
        echo "lc = $i Meshtype = $m"
        python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$m
        # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="pad" --mesh_type=$mesh_type
        # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="iso" --boundary_scheme="full" --mesh_type=$mesh_type
        # python ./script/build_helmholtz_square.py --lc=$i   --rand_seed $rand_seed --n_samples $n_samples_train --field_type "aniso" --boundary_scheme "pad" --mesh_type=$mesh_type
    done
done