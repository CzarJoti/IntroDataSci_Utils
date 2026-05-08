import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from scipy.sparse import issparse
from numpy import ndarray

symbolic = ['proto', 'service', 'state']

def split(file):
    df = pd.read_csv(file)
    X = df.drop(['id', 'attack_cat', 'label'], axis=1)
    y = df['label'].values

    return X, y

class UNSW(Dataset):
    def __init__(self, X, y, transformer=None, extracted_features = None) -> None:
        super().__init__()

        train = transformer is None
        
        self.y = y

        numeric = [c for c in X.columns if c not in symbolic]

        if extracted_features is not None:
            X = X[extracted_features]

        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = ColumnTransformer([
                    ('symbolic', OneHotEncoder(handle_unknown='ignore'), symbolic),
                    ('numeric', MinMaxScaler(), numeric)
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
