mesh_types=(6 2 0)
rand_seed=42
n_samples_train=3

for m in "${mesh_types[@]}"; do
    python build_burgers_square.py --rand_seed=$rand_seed --n_case=$n_samples_train --mesh_type=$m

    # python build_helmholtz_square.py  --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$m
    # python build_helmholtz_poly.py --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$m
    # python build_poisson_square.py --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$m
    # python build_poisson_poly.py --rand_seed=$rand_seed --n_samples=$n_samples_train --field_type="aniso" --boundary_scheme="full" --mesh_type=$m
    # python build_burgers_square.py --rand_seed=$rand_seed --n_case=$n_samples_train --mesh_type=$m
    # python build_swirl.py --lc=0.028 --alpha=1.5 --r_0 0.20 --x_0 0.3 --y_0 0.3 --n_monitor_smooth=10 --mesh_type=$m

done