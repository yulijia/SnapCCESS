import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from snapccess.model import snapshotVAE
from snapccess.train import train_model
from snapccess.util import setup_seed


## read datasets
path = '../in/'
out = '../out/'

ctfile = 'CITEseq_celltypes.csv.gz'
rnafile = 'CITEseq_logRNA.csv.gz'
adtfile = 'CITEseq_logADT.csv.gz'

celltype = pd.read_csv(path+ctfile, index_col=0)

rna = pd.read_csv(path+rnafile, index_col=0).T
rna = rna.reset_index(drop=True)
 
pro = pd.read_csv(path+adtfile, index_col=0).T
pro = pro.reset_index(drop=True)

## get the number of features
nfeatures_rna = rna.shape[1]
nfeatures_pro = pro.shape[1]


## parameters
batch_size = 64
epochs_per_cycle =1
epochs = epochs_per_cycle*100
lr = 0.02
z_dim = 100
hidden_rna2 = 185 
hidden_pro2 = 30 
feature_num = nfeatures_rna + nfeatures_pro 
## standardise each modality of the dataset
rna_sample_scaled=(pd.DataFrame(rna)-pd.DataFrame(rna).mean())/pd.DataFrame(rna).std()
pro_sample_scaled=(pd.DataFrame(pro)-pd.DataFrame(pro).mean())/pd.DataFrame(pro).std()

# combine the standardised modalities to create input data
citeseq = pd.concat([rna_sample_scaled, pro_sample_scaled], axis=1)
train_data=citeseq.to_numpy(dtype=np.float32)

# load data
train_transformed_dataset = train_data
train_dl = DataLoader(train_transformed_dataset, batch_size=batch_size,shuffle=False, num_workers=0,drop_last=False)
test_transformed_dataset = train_data
valid_dl = DataLoader(test_transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=0,drop_last=False)


## run VAE with Snapshot
model = snapshotVAE(num_features=[nfeatures_rna,nfeatures_pro], num_hidden_features=[hidden_rna2,hidden_pro2], z_dim=z_dim).cuda()


model,histroy,embedding = train_model(model, train_dl, valid_dl, lr=lr, epochs=epochs,epochs_per_cycle=epochs_per_cycle, save_path="",snapshot=True,embedding_number=1)


kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(embedding[99])
kmeans.labels_


labels_df = pd.DataFrame({'Cluster Labels': kmeans.labels_})

# Save the labels to a CSV file
labels_df.to_csv(out + 'clustering.csv', index=False)


## save all embeddings
for ind,eb in enumerate(embedding):
    eb.to_csv(out+'CITEseq'+'_embedding_{}.csv.gz'.format(ind),
        index=False,compression="gzip")