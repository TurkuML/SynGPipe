import numpy as np
import pandas as pd 
from scipy import spatial
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score

# --- Metrics Fidelity Report ---
# Total Variation Distance for continuous distributions     
def tvd_continuous(p, q, x):
    # Normalize the PDFs to ensure it integrates to 1 
    p = p / np.trapezoid(p, x) 
    q = q / np.trapezoid(q, x)    
    # Compute absolute difference
    abs_diff = np.abs(p - q)    
    # Integrate the difference and multiply by 1/2
    tvd = 0.5 * np.trapezoid(abs_diff, x)
    return tvd

def tvd_discrete(p, q):
    p = np.asarray(p)
    q = np.asarray(q)    
    #Normalize the probabilities in p and q to sum to 1   
    p = p / sum(p)
    q = q / sum(q) 
    tvd = 0.5 * np.sum(np.abs(p - q))
    return tvd
    
# Compute the Hellinger distance
def hellinger_distance(p, q):    
    p = np.asarray(p)
    q = np.asarray(q)    
    #Normalize the probabilities in p and q to sum to 1   
    p = p / sum(p)
    q = q / sum(q)    
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

   
# Compute the Jensen-Shannon distance    
def jensen_shannon_distance(p, q):
    return spatial.distance.jensenshannon(p, q, base=2) #Scipy normalized the probabilities to sum to 1

def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    #Normalize the probabilities in p and q to sum to 1  
    p = p / sum(p)
    q = q / sum(q)
    # Ensure there are no zero probabilities in q where p is non-zero to avoid division by zero
    if np.any((p != 0) & (q == 0)):
        return np.inf    
    # Compute the KL divergence
    kl_div = np.sum(p * np.log(p / q))
    return kl_div

def corr_matrix(X, method='spearman'):
    X_corr = X.corr(method=method)    
    return X_corr 

# Cramér's V function assess the strength of association
def cramers_v(confusion_matrix):
    """Returns NaN if the association is undefined (a variable with a single category, or an empty table).
    """
    r, k = confusion_matrix.shape
    n = confusion_matrix.to_numpy().sum()
    if min(r, k) < 2 or n == 0:
        return np.nan
    chi2, _, _, _ = chi2_contingency(confusion_matrix, correction=False)
    return float(np.sqrt((chi2 / n) / (min(k, r) - 1)))

def contingency_similarity(real_x, real_y, synth_x, synth_y):
    """Two-way contingency similarity: 1 - TVD between the normalized contingency tables.
    1 = identical joint distributions, 0 = no overlap.
    Categories missing from one dataset are counted as probability 0 (not ignored).
    """
    real_ct = pd.crosstab(real_x, real_y, normalize=True)
    synth_ct = pd.crosstab(synth_x, synth_y, normalize=True)

    rows = real_ct.index.union(synth_ct.index)
    cols = real_ct.columns.union(synth_ct.columns)
    real_ct = real_ct.reindex(index=rows, columns=cols, fill_value=0)
    synth_ct = synth_ct.reindex(index=rows, columns=cols, fill_value=0)

    return 1.0 - 0.5 * np.abs(real_ct.to_numpy() - synth_ct.to_numpy()).sum()

# --- Metrics Privacy Report ---
def _norm_label(value):
    """Normalize a category label so 1, 1.0 and '1' compare as equal."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(f)) if f.is_integer() else str(f)


def fit_gower_params(reference_df, cat_cols):
    """Fix the Gower settings on a reference dataset (the Real training set).
    reference_df : DataFrame used to compute the numerical ranges
    cat_cols     : columns compared as categorical (match = 0, mismatch = 1)
    Returns a dict reused by every gower_distances call, so all distances share one scale.
    """
    cat_cols = [c for c in reference_df.columns if c in set(cat_cols)]
    num_cols = [c for c in reference_df.columns if c not in set(cat_cols)]
    ranges = np.ones(len(num_cols))
    for j, c in enumerate(num_cols):
        col = pd.to_numeric(reference_df[c], errors="coerce")
        r = float(col.max() - col.min())
        # Constant column: any difference counts as a full mismatch (after capping at 1)
        ranges[j] = r if r > 0 else np.finfo(float).eps
    return {"num_cols": num_cols, "cat_cols": cat_cols, "ranges": ranges}


def gower_distances(df1, df2=None, params=None):
    """Gower distance matrix between the records of df1 and df2, using FIXED settings.
    params : output of fit_gower_params (ranges from the training set). If None, the
             settings are fitted on df1 (only for standalone use, not for the report).
    Numerical differences are |x - y| / range, capped at 1. Missing values count as 1.
    If df2 is None, distances within df1 are computed and the diagonal is set to inf.
    Returns (distances, index of the closest record in df2 for each record in df1).
    """
    self_compare = df2 is None
    if self_compare:
        df2 = df1
    if params is None:
        params = fit_gower_params(df1, [c for c in df1.columns
                                        if not pd.api.types.is_numeric_dtype(df1[c])])

    num_cols, cat_cols, ranges = params["num_cols"], params["cat_cols"], params["ranges"]
    n1, n2 = len(df1), len(df2)
    n_features = len(num_cols) + len(cat_cols)
    dist = np.zeros((n1, n2), dtype=np.float64)

    # Numerical features: one column at a time to keep memory at n1 x n2
    for j, c in enumerate(num_cols):
        x = pd.to_numeric(df1[c], errors="coerce").to_numpy(dtype=float) / ranges[j]
        y = pd.to_numeric(df2[c], errors="coerce").to_numpy(dtype=float) / ranges[j]
        d = np.abs(x[:, None] - y[None, :])
        d = np.where(np.isnan(d), 1.0, np.minimum(d, 1.0))
        dist += d

    # Categorical features: shared integer codes so equal labels match exactly
    for c in cat_cols:
        labels = pd.concat([df1[c], df2[c]], ignore_index=True).map(_norm_label)
        codes, _ = pd.factorize(labels)
        dist += (codes[:n1, None] != codes[None, n1:]).astype(np.float64)

    dist /= n_features
    if self_compare:
        np.fill_diagonal(dist, np.inf)
    closest = np.argmin(dist, axis=1)
    return dist, closest

# Compute Precision@K for membership inference
def precision_at_k_low_scores(y_true, y_scores, k):
    """Expected Precision@K when ranking by ascending score (lowest distance = most likely member).
    Ties are resolved as if broken uniformly at random, so the result does not depend
    on the order of the records (members first / non-members first).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    k = int(min(k, len(y_scores)))
    if k <= 0:
        return np.nan
    t = np.sort(y_scores)[k - 1] # score of the K-th ranked record
    below = y_scores < t # always inside the top K
    tied = y_scores == t # compete for the remaining slots
   
    n_below = below.sum()
    slots = k - n_below # how many tied records enter the top K
    expected_members = y_true[below].sum() + slots * y_true[tied].mean()
    return expected_members / k

# --- Metrics for Utility Report ---

def align_proba(proba, model_classes, n_classes):
    """
    Place predicted probabilities into a matrix with one column per encoded class
    (0..n_classes-1). Classes the model never saw during training get probability 0.

    proba         : output of model.predict_proba, shape (n_samples, len(model_classes))
    model_classes : model.classes_ (encoded integer labels)
    n_classes     : total number of classes across all datasets
    """
    proba = np.asarray(proba, dtype=float)
    aligned = np.zeros((proba.shape[0], n_classes))
    aligned[:, np.asarray(model_classes, dtype=int)] = proba
    return aligned

def regression_metrics(y_true, y_pred):
    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }

def classification_metrics(y_true, y_pred_proba, binary=False):
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba)    
  
    if binary:
        avg = "binary"
        if y_pred_proba.ndim == 2:
            y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)   # probability of class 1
        else:
            y_pred = (y_pred_proba >= 0.5).astype(int)
    else:
        avg = "macro"
        y_pred = np.argmax(y_pred_proba, axis=1)  # convert probs → class labels

    # Ensure predicted labels are 1D
    y_pred = np.asarray(y_pred).ravel()
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }
    
    if binary:
        metrics["AUC"] = roc_auc_score(y_true, y_pred_proba[:, 1])
        display_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    else:
        display_labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    
    return metrics, display_labels
