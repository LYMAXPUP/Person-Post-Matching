import numpy as np
import scipy.stats


def l2_normalize(vecs):
    """向量标准化

    Parameters:
    ------------
    vecs : list
        形如[[1,2,3], [4,5,6], ...]
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def l2_normalize_2(vec):
    """向量标准化

    Parameters:
    -------------
    vec : list
        形如[1,2,3]
    """
    norms = (vec ** 2).sum() ** 0.5
    return vec / np.clip(norms, 1e-8, np.inf)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation




