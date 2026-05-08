import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from scipy.sparse import issparse
from numpy import ndarray

class UNSW(Dataset):
    def __init__(self, file, transformer=None, extract_features=False) -> None:
        super().__init__()

        train = transformer is None

        df = pd.read_csv(file)
        X = df.drop(['id', 'attack_cat', 'label'], axis=1)
        self.y = df['label'].values

        symbolic = ['proto', 'service', 'state']
        numeric = [c for c in X.columns if c not in symbolic]

        if extract_features:
            ex_X = X.copy()
            for s in symbolic:
                ex_X[s] = LabelEncoder().fit_transform(ex_X[s]) # type: ignore

            discrete_features = [col in symbolic for col in ex_X.columns]
            percent = SelectPercentile(
                score_func=lambda X, y: mutual_info_classif(X, y, discrete_features=discrete_features),
                percentile=50
            )

            p = percent.fit(X, self.y)  # type: ignore
            features = ex_X.columns[p.get_support()]
            
            X = X[features]

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
