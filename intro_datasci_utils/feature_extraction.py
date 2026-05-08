from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.preprocessing import LabelEncoder

def extract_features(X, y, symbolic):
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

            return features