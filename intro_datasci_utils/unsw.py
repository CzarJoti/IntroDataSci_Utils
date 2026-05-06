import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import issparse
from numpy import ndarray

class UNSW(Dataset):
    testing_url = "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/download.aspx?UniqueId=2a810f6a%2Dcc3d%2D4d98%2D909e%2D37489d8daf98"
    training_url = "https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/download.aspx?UniqueId=49413d38%2D3330%2D4358%2Dbfa2%2D0349031198a5"

    def __init__(self, transformer=None) -> None:
        super().__init__()
        url: str

        train = transformer is None
        if train:
            url = self.training_url
        else:
            url = self.testing_url

        df = pd.read_csv(url)
        X = df.drop(['id', 'attack_cat', 'label'], axis=1)
        self.y = df['label'].values

        symbolic = ['proto', 'service', 'state']
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
        return(torch.tensor(self.X[idx], dtype=torch.float32))
