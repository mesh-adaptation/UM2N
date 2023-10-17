# Author: Chunyang Wang
# GitHub Username: chunyang-w

# Functionality: Plot the training loss and tangled element per mesh for a single training run  # noqa
#                Two curve, ie. training loss and validation loss, are plotted in the same figure  # noqa

# %% import packages and setup
import pandas as pd
import numpy as np  # noqa
import matplotlib.pyplot as plt  # noqa

# %% plot model training loss, test tangle
model_name = "M2N_orignal"
loss_path = "/Users/cw1722/Downloads/M2N_og_20_iso/train_log/loss.csv" # noqa
tangle_path = "/Users/cw1722/Downloads/M2N_og_20_iso/train_log/tangle.csv" # noqa

loss_data = pd.read_csv(loss_path)
tangle_data = pd.read_csv(tangle_path)

# Filter out the data points with loss > 50 and tangle > 10
loss_data_filtered = loss_data[
    (loss_data['Train Loss'] <= 50) & (loss_data['Test Loss'] <= 50)]
tangle_data_filtered = tangle_data[
    tangle_data['Test Tangle'] <= 10]

# Combined subplot with loss on the left and tangle on the right
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(48, 14))

# Left subplot: Loss plot
final_epoch = loss_data_filtered['Epoch'].iloc[-1]
final_test_loss = loss_data_filtered['Test Loss'].iloc[-1]
final_train_loss = loss_data_filtered['Train Loss'].iloc[-1]

ax1.plot(
    loss_data_filtered['Epoch'],
    loss_data_filtered['Train Loss'],
    label=f'{model_name} Train Loss, final: {final_train_loss:.3f}',
    color='cyan', lw=2, alpha=0.65)

ax1.plot(loss_data_filtered['Epoch'],
         loss_data_filtered['Test Loss'],
         label=f'{model_name} Test Loss, final: {final_test_loss:.3f}',
         color='blue', lw=2, alpha=0.65)


# right subplot: Tangle plot
final_test_tangle = tangle_data_filtered['Test Tangle'].iloc[-1]
final_train_tangle = tangle_data_filtered['Train Tangle'].iloc[-1]

ax2.plot(
    tangle_data_filtered['Epoch'],
    tangle_data_filtered['Train Tangle'],
    label=f'{model_name} Train Tangle, final: {final_train_tangle:.3f}',
    color='cyan', lw=2, alpha=0.65)
# tangled element per mesh on test set
ax2.plot(
    tangle_data_filtered['Epoch'],
    tangle_data_filtered['Test Tangle'],
    label=f'{model_name} Test Tangle, final: {final_test_tangle:.3f}',
    color='blue', lw=2, alpha=0.65)

# make a red dot where all the tangle is zero (both train and test)
no_train_tangle_idx = tangle_data_filtered[
    (tangle_data_filtered['Train Tangle'] == 0)
].index

no_test_tangle_idx = tangle_data_filtered[
    (tangle_data_filtered['Test Tangle'] == 0)
].index

no_tangle_idx = tangle_data_filtered[
    (tangle_data_filtered['Train Tangle'] == 0) &
    (tangle_data_filtered['Test Tangle'] == 0)
].index

epsilon = 0.008  # move the dot up a little bit so it's not covered by the axis
ax2.scatter(
    tangle_data_filtered["Epoch"].iloc[no_train_tangle_idx],
    tangle_data_filtered['Train Tangle'].iloc[no_train_tangle_idx] + epsilon,
    color='cyan', edgecolors='black',
    linewidths=1.2, s=40, zorder=6,
    label=f'No train tangle({len(no_train_tangle_idx)})')

ax2.scatter(
    tangle_data_filtered["Epoch"].iloc[no_test_tangle_idx],
    tangle_data_filtered['Test Tangle'].iloc[no_test_tangle_idx] + epsilon,
    color='blue', edgecolors='black',
    linewidths=1.2, s=40, zorder=5,
    label=f'No test tangle({len(no_test_tangle_idx)})')

ax2.scatter(
    tangle_data_filtered["Epoch"].iloc[no_tangle_idx],
    tangle_data_filtered['Train Tangle'].iloc[no_tangle_idx] + epsilon,
    color='red', edgecolors='black', marker='P',
    linewidths=0.8, s=80, zorder=7,
    label=f'No tangle({len(no_tangle_idx)})')

# Adjust the layout and display the combined subplot
ax1.set_xlabel('Epoch', fontsize=22)
ax1.set_ylabel('Loss', fontsize=22)
ax1.legend(loc='upper right', fontsize=20)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='both', which='major', labelsize=20)

ax2.set_ylim(0, 1)
ax2.set_xlabel('Epoch', fontsize=22)
ax2.set_ylabel('Tangle', fontsize=22)
ax2.legend(loc='upper right', fontsize=20)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='both', which='major', labelsize=20)

fig.suptitle(f'{model_name} Training Loss and Tangle', fontsize=24)
plt.tight_layout()
plt.show()

# %%
