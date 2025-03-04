# %% [markdown]
# NSD fMRI regression performance\
# cliptext 1st embedding:\
# corr: `0.9999999999999798` euclidian dist: `0.08716399866359316`\
# clipvision 1st embedding:\
# corr: `0.7206470311671278` euclidian dist: `0.7445045003979572`\
# autokl:\
# corr: `0.021463346119866764` euclidian dist: `113.58405093418304`\
# \
# THINGS-MEG unmodified regression performance\
# cliptext 1st embedding: \
# corr: `0.7132301973694442` euclidian dist: `0.6739413756833288` \
# clipvision 1st embedding: \
# corr: `0.6127160358947972` euclidian dist: `0.878658145980226` \
# autokl:\
# corr: `0.005573586983020162` euclidian dist: `114.27715624831653`

# %%
from models import SimpleConv, BrainModule
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from torch import nn, optim
import numpy as np
from collections import namedtuple
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import correlation
import os

# %%
data_array = np.load('cache/processed_data/BIGMEG1/train_thingsmeg_sub-BIGMEG1.npy', mmap_mode='r')
# labels_array = np.load('cache/extracted_embeddings/BIGMEG1/train_cliptext1b_sub-BIGMEG1.npy', mmap_mode='r')
# labels_array = np.load('cache/extracted_embeddings/BIGMEG1/train_cliptext1bcategory_sub-BIGMEG1.npy', mmap_mode='r')
labels_array = np.load('cache/extracted_embeddings/BIGMEG1/train_clipvision1b_sub-BIGMEG1.npy', mmap_mode='r')
# labels_array = labels_array[:, 6]
print(data_array.shape, labels_array.shape)

# %%
# device = torch.device('cuda:7')
# model = BrainModule()
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# model.to(device);

# %%
# # Optional: see network parameters
# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Number of parameters: {param.numel()}")
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params}")

# %%
models = []

# %%
subject = 'BIGMEG1'
# save_dir = 'cache/cliptext1b_module_weights/' + subject + '/'
# save_dir = 'cache/cliptext1bcategory_module_weights/' + subject + '/'
save_dir = 'cache/clipvision1b_module_weights/' + subject + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# for i_token in range(77): # cliptext
for i_token in range(257): # clipvision
    device = torch.device('cuda:7')
    model = BrainModule()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.to(device);

    data_tensor = torch.tensor(data_array).float()
    labels_tensor = torch.tensor(labels_array[:, i_token]).float()
    # Create a TensorDataset from your data tensor and labels tensor
    dataset = TensorDataset(data_tensor, labels_tensor)

    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_data, batch_size=32)
    val_dataloader = DataLoader(val_data, batch_size=32)

    # Training loop
    for epoch in range(5):  # 100 epochs
        # Training phase
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            val_loss = 0
            average_euclidean_distance = 0
            correlations = 0
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Compute the Euclidean distances
                euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(outputs.cpu(), labels.cpu())])
                correlation_distances = np.array([correlation(u, v) for u, v in zip(outputs.cpu(), labels.cpu())])
                # Compute the average Euclidean distance
                average_euclidean_distance += euclidean_distances.mean()
                correlations += (1 - correlation_distances).mean()
                
            val_loss /= len(val_dataloader)
            average_euclidean_distance /= len(val_dataloader)
            correlations /= len(val_dataloader)

        model.train()  # Set the model back to training mode

        # clear_output(wait=False)
        print(f'Token {i_token}, Epoch {epoch+1} / {5}, Training Loss: {loss.item()}, Validation Loss: {val_loss}, Average Euclidean Distance: {average_euclidean_distance}, Correlations: {correlations}')
    
    models.append(model)
    torch.save(model.state_dict(), save_dir + f'{i_token}.pth')

# # %%

# save_file = save_dir + f'thingsmeg_dummymodule_cliptext1b_weights_sub-{subject}.pth'
# torch.save(model.state_dict(), save_file)

# # %%
# test_data = np.load('cache/processed_data/BIGMEG1/test_thingsmeg_sub-BIGMEG1.npy', mmap_mode='r')
# test_labels = np.load('cache/extracted_embeddings/BIGMEG1/test_cliptext1b_sub-BIGMEG1.npy', mmap_mode='r')
# # test_labels = np.load('cache/extracted_embeddings/BIGMEG1/test_clipvision1b_sub-BIGMEG1.npy', mmap_mode='r')
# test_labels = test_labels[:, 0]

# test_data_tensor = torch.tensor(test_data).float()
# test_labels_tensor = torch.tensor(test_labels).float()

# test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
# test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# # Testing loop
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():  # Disable gradient calculation
#     preds = []
#     for inputs, labels in test_dataloader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         preds.extend(outputs.cpu().numpy())
#     pred_labels = np.array(preds)
# # Compute the Euclidean and correlation distances
# euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_labels, test_labels)])
# correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_labels, test_labels)])
# # Compute the average Euclidean distance
# average_euclidean_distance = euclidean_distances.mean()
# correlations = (1 - correlation_distances).mean()
# print('corr:', correlations, 'euclidian dist:' ,average_euclidean_distance)

# # %%
# # corr: 0.8331517203063764 euclidian dist: 0.4900515302499065

# # %%
# # # save pred cliptext
# # preds_array = np.array(preds)
# # preds_repeated = np.repeat(preds_array[:, np.newaxis, :], 77, axis=1)
# # # np.save('cache/predicted_embeddings/BIGMEG1/thingsmeg_dummymodule_cliptext1b_sub-BIGMEG1.npy', preds_repeated)

# # # save pred clipvision
# # preds_array = np.array(preds)
# # regress_labels = np.load('cache/predicted_embeddings/BIGMEG1/thingsmeg_regress_clipvision1b_sub-BIGMEG1.npy')
# # regress_labels[:, 0] = preds_array
# # # np.save('cache/predicted_embeddings/BIGMEG1/thingsmeg_dummyregress_clipvision1b_sub-BIGMEG1.npy', regress_labels)

# # %% [markdown]
# # ## reference metrics

# # %%
# # NSD fMRI regression performance
# test_labels = np.load('/mnt/sphere/projects/brain-diffuser/data/extracted_features/subj01/nsd_cliptext_test.npy', mmap_mode='r')
# test_labels = test_labels[:, 0, :]
# pred_labels = np.load('/mnt/sphere/projects/brain-diffuser/data/predicted_features/subj01/nsd_cliptext_predtest_nsdgeneral.npy', mmap_mode='r')
# pred_labels = pred_labels[:, 0, :]
# # Compute the Euclidean distances
# euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_labels, test_labels)])
# correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_labels, test_labels)])
# # Compute the average Euclidean distance
# average_euclidean_distance = euclidean_distances.mean()
# correlations = (1 - correlation_distances).mean()
# print('cliptext 1st embedding:')
# print('corr:', correlations, 'euclidian dist:', average_euclidean_distance)
# test_labels = np.load('/mnt/sphere/projects/brain-diffuser/data/extracted_features/subj01/nsd_clipvision_test.npy', mmap_mode='r')
# test_labels = test_labels[:, 0, :]
# pred_labels = np.load('/mnt/sphere/projects/brain-diffuser/data/predicted_features/subj01/nsd_clipvision_predtest_nsdgeneral.npy', mmap_mode='r')
# pred_labels = pred_labels[:, 0, :]
# # Compute the Euclidean distances
# euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_labels, test_labels)])
# correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_labels, test_labels)])
# # Compute the average Euclidean distance
# average_euclidean_distance = euclidean_distances.mean()
# correlations = (1 - correlation_distances).mean()
# print('clipvision 1st embedding:')
# print('corr:', correlations, 'euclidian dist:', average_euclidean_distance)
# test_labels = np.load('/mnt/sphere/projects/brain-diffuser/data/extracted_features/subj01/nsd_vdvae_features_31l.npz', mmap_mode='r')
# test_labels = test_labels['test_latents']
# pred_labels = np.load('/mnt/sphere/projects/brain-diffuser/data/predicted_features/subj01/nsd_vdvae_nsdgeneral_pred_sub1_31l_alpha50k.npy', mmap_mode='r')
# # Compute the Euclidean distances
# euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_labels, test_labels)])
# correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_labels, test_labels)])
# # Compute the average Euclidean distance
# average_euclidean_distance = euclidean_distances.mean()
# correlations = (1 - correlation_distances).mean()
# print('autokl:')
# print('corr:', correlations, 'euclidian dist:', average_euclidean_distance)

# # %%
# # THINGS-MEG unmodified regression performance
# test_labels = np.load('cache/extracted_embeddings/BIGMEG1/test_cliptext1b_sub-BIGMEG1.npy', mmap_mode='r')
# test_labels = test_labels[:, 0, :]
# pred_labels = np.load('cache/predicted_embeddings/BIGMEG1/thingsmeg_regress_cliptext1b_sub-BIGMEG1.npy', mmap_mode='r')
# pred_labels = pred_labels[:, 0, :]
# # Compute the Euclidean distances
# euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_labels, test_labels)])
# correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_labels, test_labels)])
# # Compute the average Euclidean distance
# average_euclidean_distance = euclidean_distances.mean()
# correlations = (1 - correlation_distances).mean()
# print('cliptext 1st embedding:')
# print(correlations, average_euclidean_distance)
# test_labels = np.load('cache/extracted_embeddings/BIGMEG1/test_clipvision1b_sub-BIGMEG1.npy', mmap_mode='r')
# test_labels = test_labels[:, 0, :]
# pred_labels = np.load('cache/predicted_embeddings/BIGMEG1/thingsmeg_regress_clipvision1b_sub-BIGMEG1.npy', mmap_mode='r')
# pred_labels = pred_labels[:, 0, :]
# # Compute the Euclidean distances
# euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_labels, test_labels)])
# correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_labels, test_labels)])
# # Compute the average Euclidean distance
# average_euclidean_distance = euclidean_distances.mean()
# correlations = (1 - correlation_distances).mean()
# print('clipvision 1st embedding:')
# print(correlations, average_euclidean_distance)
# test_labels = np.load('cache/extracted_embeddings/BIGMEG1/test_autokl1b_sub-BIGMEG1.npy', mmap_mode='r')
# pred_labels = np.load('cache/predicted_embeddings/BIGMEG1/thingsmeg_regress_autokl1b_sub-BIGMEG1.npy', mmap_mode='r')
# # Compute the Euclidean distances
# euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_labels, test_labels)])
# correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_labels, test_labels)])
# # Compute the average Euclidean distance
# average_euclidean_distance = euclidean_distances.mean()
# correlations = (1 - correlation_distances).mean()
# print('autokl:')
# print('corr:', correlations, 'euclidian dist:', average_euclidean_distance)

# # %%



