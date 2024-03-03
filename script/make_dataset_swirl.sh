###################### Test dataset generation ######################
# Test data
# Problem type: Swirl
# Meshtype: 0, 2, 6
# n_samples: 100
# Random_seed: 42
# Resolution: 0.05, 0.028

rand_seed=42
mesh_types=(6 2 0)
monitor_smooths=(10 15 20 30 40)
for m in "${mesh_types[@]}"; do
    for m_s in "${monitor_smooths[@]}"; do
        echo "Swirl meshtype = $m monitor smooth = $m_s"
        python ./script/build_swirl.py --lc=0.028 --alpha=1.5 --r_0 0.20 --x_0 0.25 --y_0 0.25 --mesh_type=$m --n_monitor_smooth=$m_s
    done
done

# for m in "${mesh_types[@]}"; do
#     python ./script/build_swirl.py --lc=0.05 --alpha=1.5 --r_0 0.20 --x_0 0.25 --y_0 0.25 --mesh_type=$m
# done


