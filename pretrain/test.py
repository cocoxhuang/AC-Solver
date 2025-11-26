from pretrain.data.dataset import DualRingActor_dataset, FFN_dataset

data_path = "pretrain/data/gs_sa.csv"
# dataset = DualRingActor_dataset(data_path)
dataset = FFN_dataset(data_path)
print(dataset.dataloaders()[0].dataset.tensors[0][0])
print(dataset._n_actions)
