from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.preprocessing import LabelEncoder

def extract_features(X, y, symbolic, percentage):
            ex_X = X.copy()
            for s in symbolic:
                ex_X[s] = LabelEncoder().fit_transform(ex_X[s]) # type: ignore

            discrete_features = [col in symbolic for col in ex_X.columns]
            percent = SelectPercentile(
                score_func=lambda X, y: mutual_info_classif(X, y, discrete_features=discrete_features),
                percentile=percentage
            )

            p = percent.fit(ex_X, y)  # type: ignore
            features = p.get_feature_names_out()

            importances = mutual_info_classif(ex_X, y, discrete_features=discrete_features)

            return features, importances, p.feature_names_in_