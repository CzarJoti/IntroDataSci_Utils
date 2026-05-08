import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import issparse
from numpy import ndarray

class NLS_KDD(Dataset):
    def __init__(self, file: str, transformer=None) -> None:
        super().__init__()

        train = transformer is None

        df = pd.read_csv(file)
        X = df.drop(['class'], axis=1)

        mapping = {'normal': 0, 'anomaly': 1}
        self.y = df['label'].map(mapping).values

        symbolic = ['protocol_type', 'service', 'flag']
        numeric = [c for c in X.columns if c not in symbolic]

        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = ColumnTransformer([
                    ('symbolic', OneHotEncoder(handle_unknown='ignore'), symbolic),
                    ('numeric', StandardScaler(), numeric)
            ], sparse_threshold=0)

        tX  = self.transformer.fit_transform(X) if train else self.transformer.transform(X)
        
        self.X: ndarray
        if issparse(tX):
            self.X = tX.todense() # type: ignore
        else:
            self.X = tX # type: ignore
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32)), (torch.tensor(self.y[idx], dtype=torch.float32))


