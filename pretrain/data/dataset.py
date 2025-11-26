import pandas as pd
from data.tokenizer import tokenizer, FFN_tokenizer, DualRingActor_tokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch

from abc import ABC, abstractmethod

class BaseDataset(ABC):
    def __init__(self, data_path: str):
        '''
        Args:
            data_path: Path to the txt data file
        '''
        if not isinstance(data_path, str):
            raise TypeError("data_path must be a string")
        # if not data_path.endswith('.txt'):
        #     raise ValueError("data_path must be a .txt file")
        
        self._data_path = data_path

        # Load the data from the file
        try:
            self._data = pd.read_csv(self._data_path, sep=",", names=["r1","r2", "x", "k", "l", "inverse"], index_col=False)
        except Exception as e:
            raise Exception(f"Data file could not be read. Please check the file format. Error: {e}")
        
        self._max_relator_length = max(self._data['r1'].apply(lambda s: len(s)).max(), self._data['r2'].apply(lambda s: len(s)).max())

        self._tokenizer = None
        self._n_actions = None
        self._processed = False  # Flag to track if data has been processed

    @property
    def max_relator_length(self) -> int:
        return self._max_relator_length
    
    @property
    def tokenizer(self) -> tokenizer:
        return self._tokenizer

    def process_data(self):
        '''Process the raw data to tokenize relators r1 and r2 individually,
        combine them into states, and create target labels.'''
        if self._processed:
            return
            
        self._data['r1'] = self._data['r1'].apply(lambda s: self._tokenizer.tokenize(s))
        self._data['r2'] = self._data['r2'].apply(lambda s: self._tokenizer.tokenize(s))
        self._data["state"] = self._data['r1'].apply(lambda x: tuple(x)) + self._data['r2'].apply(lambda x: tuple(x))
        self._data["state"] = self._data["state"].apply(lambda x: self._tokenizer.padding(x, self._max_relator_length*2))

        self._data['action'] = self._data.apply(lambda row: row['x'] + "," + str(row['k']) + "," + str(row['l']) + "," + str(row['inverse']), axis=1)

        # # self._data['action'] = self._data['action'].apply(lambda action_str: self._tokenizer._substitution_dict_inv[action_str])
        # self._data['action'] = self._data['action'].map(self._tokenizer._substitution_dict_inv)

        # give an one-hot encoding to actions by their frequency order
        action_order = self._data['action'].value_counts().index.tolist()
        action_mapping = {action: idx for idx, action in enumerate(action_order)}
        self._data['action'] = self._data['action'].map(action_mapping)

        # # filter to only the top 20 actions
        self._data = self._data[self._data['action'] < 20]

        self._data = self._data[['state', 'action']]

        # self._n_actions = self._tokenizer._substitution_dict.__len__()
        self._n_actions = self._data['action'].nunique()

        self._processed = True

    @property
    def n_actions(self) -> int:
        '''Return the number of unique actions in the dataset.'''
        if self._n_actions is None:
            raise ValueError("Data has not been processed yet. Call process_data() before accessing n_actions.")
        else:
            return self._n_actions
        
    def tensorDataset(self) -> TensorDataset:
        '''Create a TensorDataset for the entire dataset.'''
        self.process_data()

        X = self._data['state'].tolist()
        y = self._data['action'].tolist()

        X_tensor = torch.tensor(X, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        return dataset

    def dataloaders(self, batch_size: int = 32,
                seed: int = 42) -> list:
        '''Create train and eval dataloaders for the dataset.'''
        torch.manual_seed(seed)

        dataset = self.tensorDataset()
        X_tensor, y_tensor = dataset.tensors[0], dataset.tensors[1]
        
        perm_ids = torch.randperm(len(X_tensor))
        train_ids, eval_ids = perm_ids[:int(0.8*len(perm_ids))], perm_ids[int(0.8*len(perm_ids)):]
        X_train, y_train, X_eval, y_eval = X_tensor[train_ids], y_tensor[train_ids], X_tensor[eval_ids], y_tensor[eval_ids]

        train_dataset = TensorDataset(X_train, y_train)
        eval_dataset = TensorDataset(X_eval, y_eval)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        return [train_dataloader, eval_dataloader]

class FFN_dataset(BaseDataset):
    def __init__(self, data_path: str):
        super().__init__(data_path) 
        self._tokenizer = FFN_tokenizer(self._max_relator_length)

    def tensorDataset(self):
        dataset = super().tensorDataset()
        return TensorDataset(dataset.tensors[0].float(), dataset.tensors[1])
    
class DualRingActor_dataset(BaseDataset):
    def __init__(self, data_path: str):
        super().__init__(data_path) 
        self._tokenizer = DualRingActor_tokenizer(self._max_relator_length)

if __name__ == "__main__":
    data_path = "data/gs_sa.csv"
    # dataset = DualRingActor_dataset(data_path)
    dataset = FFN_dataset(data_path)
    print(dataset.dataloaders()[0].dataset.tensors[0][0])