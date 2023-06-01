import random
import math
import numpy as np
import itertools
import scipy.stats as stats
from heapdict import heapdict
import operator
from collections import Counter
import heapq
from sklearn import metrics
import itertools
import multiprocessing as mp


# ----------------------------------------------------------------------------------------------
# manipulate dataset


def generate_similarity_scores(df, pairs=None, method="l2"):
    """
    Computes pairwise similarity scores between images in a dictionary of image features.

    Args:
        df (dict): A dictionary of image features. The keys of the dictionary are identity labels, and the
        values are dictionaries of image features. The keys of the inner dictionaries are image
        labels, and the values are feature vectors for the corresponding images.
        pairs (list of tuples, optional): A list of image pairs to compute the similarity scores 
        for. If None, the function computes the pairwise similarity scores 
        for all possible image pairs within each identity and across all identities.
        method (str): The method to use for computing similarity scores. It can be 'l2' for L2 distance (output is 1/L2 distance) or
        'cosine' for cosine similarity.

    Returns:
        dict: A dictionary containing pairwise 1 / L2 distance or cosine similarity between images. The keys of the dictionary
        are identity labels, and the values are dictionaries of similarity scores to other identities.
        The keys of the inner dictionaries are identity labels, and the values are lists of similarity scores or similarities
        between all corresponding image pairs of the two identities.
    """
    if method not in ("l2", "cosine"):
        raise ValueError("method should be either 'l2' or 'cosine'")

    sim_dict = {id: {} for id in df}
    if pairs is None:
        for id1 in df:
            images1 = df[id1]
            for id2 in df:
                images2 = df[id2]
                sim_list = []
                if id1 == id2:
                    combs = itertools.combinations(images1, 2)
                else:
                    combs = itertools.product(images1, images2)
                for img1, img2 in combs:
                    if (id1 == id2 and img1 != img2) or id1 != id2:
                        if method == "l2":
                            score = 1 / math.dist(images1[img1], images2[img2])
                        else:
                            score = np.dot(images1[img1], images2[img2]) / (
                                np.linalg.norm(images1[img1]) * np.linalg.norm(images2[img2]))
                        sim_list.append(score)
                sim_dict[id1][id2] = sim_list
                sim_dict[id2][id1] = sim_list
    else:
        # Compute similarities for specified image pairs
        for id1, img1, id2, img2 in pairs:
            if method == "l2":
                score = 1 / math.dist(df[id1][img1], df[id2][img2])
            else:
                score = np.dot(df[id1][img1], df[id2][img2]) / (
                    np.linalg.norm(df[id1][img1]) * np.linalg.norm(df[id2][img2]))
            sim_dict[id1].setdefault(id2, []).append(score)
            sim_dict[id2].setdefault(id1, []).append(score)

    return sim_dict


def compute_error_metrics(df):
    """
    Computes the false negative and false positive rates for a binary classification problem.

    Args:
        df (dict): A dictionary where the keys represent the IDs of the individuals being compared, and the values are dictionaries with keys representing the IDs of the individuals they are being compared to, and the values are lists of 0s and 1s, where 1 indicates an error and 0 a non error.

    Returns:
        tuple: A tuple of two floats representing the false non-match rate (fnr) and false match rate (fpr) respectively.
    """
    neg = 0
    fp = 0
    pos = 0
    fn = 0
    for id1, inner_dict in df.items():
        for id2, values in inner_dict.items():
            values_sum = sum(values)
            if id1 == id2:
                fn += values_sum
                pos += len(values)
            else:
                fp += values_sum
                neg += len(values)
    if neg > 0:
        fpr = fp / neg
    else:
        fpr = np.nan
    if pos > 0:
        fnr = fn / pos
    else:
        fnr = np.nan
    return fnr, fpr


def generate_errors(df, threshold):
    """
    Computes errors for a pairwise biometric verification system.

    Args:
        df (dict): A dictionary where the keys represent the IDs of the individuals being compared, and the values are dictionaries with keys representing the IDs of the individuals they are being compared to, and the values are lists of floats representing the similarity scores between the two individuals.
        threshold (float): A threshold value for separating matches from non-matches. Values above the threshold are considered matches, and values below the threshold are considered non-matches.

    Returns:
        dict: A dictionary where the keys represent the IDs of the individuals being compared, and the values are dictionaries with keys representing the IDs of the individuals they are being compared to, and the values are lists of binary values (0 or 1) representing the error and non errors.
    """
    df_error = {id: {} for id in df.keys()}
    for id1 in df.keys():
        for id2 in df[id1].keys():
            values = df[id1][id2]
            df_error[id1][id2] = [1 if (v < threshold and id1 == id2) or (
                v > threshold and id1 != id2) else 0 for v in values]
    return df_error


def get_gen_and_imp_scores(df, ids_sampled=None):
    """
    Obtain list of genuine and impostor scores from a dictionary of similarity scores.

    Args:
        df (dict): A dictionary where the keys represent the IDs of the individuals being compared, and the values are dictionaries with keys representing the IDs of the individuals they are being compared to, and the values are lists of floats representing the similarity scores between the two individuals.
        ids_sampled (optional, dict): a dictionary with the identifiers that are allowed to be sampled in the calculation of the impostor scores. If None, all the identifiers will be used.

    Returns:
        list, list: List containing genuine scores and impostor scores.
    """
    if not ids_sampled:
        ids_sampled = {id: 1 for id in df.keys()}
    impostor_scores = [df[id1][id2] * ids_sampled[id2] * ids_sampled[id1]
                       for id1 in df.keys() if id1 in ids_sampled
                       for id2 in df[id1] if id1 != id2 and id2 in ids_sampled]
    genuine_scores = [df[id1][id1] * ids_sampled[id1]
                      for id1 in df.keys() if id1 in ids_sampled and id1 in df[id1]]
    return list(itertools.chain.from_iterable(genuine_scores)), list(itertools.chain.from_iterable(impostor_scores))


# ----------------------------------------------------------------------------------------------
# methods


def correct_boot_cis_by_wilson(df, fnr_ci, fpr_ci, alpha=0.05):
    """
    Correct bootstrap confidence intervals that are degenerate at either 0 or 1 by applying Wilson under independence.

     Args:
         df (dict): A dictionary where the keys represent the IDs of the individuals being compared,
         and the values are dictionaries with keys representing the IDs of the individuals they are
         being compared to, and the values are lists of 0s and 1s, where 1 indicates an error and 0
         a non error.
         fnr_ci (tuple): A tuple of two floats representing the lower and upper bounds of the confidence interval for the fnr.
         fpr_ci (tuple): A tuple of two floats representing the lower and upper bounds of the confidence interval for the fpr.

     Returns:
         tuple: A tuple of two lists representing the lower and upper
                bounds of the confidence intervals for the fnr and fpr.
     """
    if fnr_ci[1] == 0 or fnr_ci[0] == 1:
        pos = sum([len(df[id][id]) for id in df if id in df[id]])
        fn = sum([sum(df[id][id]) for id in df if id in df[id]])
        fnr_ci = get_ci_wilson(
            n_errors=fn, n_instances=pos, mean_variance=0, alpha=alpha)

    if fpr_ci[1] == 0 or fpr_ci[0] == 1:
        neg = sum(len(df[id1][id2])
                  for id1 in df for id2 in df[id1] if id1 != id2)
        fp = sum(sum(df[id1][id2])
                 for id1 in df for id2 in df[id1] if id1 != id2)
        fpr_ci = get_ci_wilson(
            n_errors=fp / 2, n_instances=neg / 2, mean_variance=1, alpha=alpha)

    return fnr_ci, fpr_ci


def compute_error_metrics_dbn_boot(df):
    """Computes the fnr and fpr for the double-or-nothing bootstrap.

    Parameters:
        df (dict): A dictionary of dictionaries, where the outer dictionary keys are identifiers for different entities, and the inner dictionary keys are the identifiers for the same entities, and the corresponding values are lists of binary labels (1 is error, 0 is correct prediction).

    Returns:
        tuple: A tuple of two floats representing the fnr and fpr.
    """
    ids = list(df.keys())
    weights = random.choices([0, 1], k=len(ids))
    ids_sampled = {ids[i]: 2 for i in range(len(ids)) if weights[i] == 1}

    fnr_boot = sum([sum(df[id][id]) * ids_sampled[id] for id in ids_sampled if id in df[id]]) / \
        sum(len(df[id][id]) * ids_sampled[id]
            for id in ids_sampled if id in df[id])
    neg = 0
    fp = 0
    for id1 in ids_sampled:
        fp_id1 = 0
        neg_id1 = 0
        for id2 in df[id1]:
            if id2 in ids_sampled and id2 != id1:
                fp_id1 += sum(df[id1][id2]) * ids_sampled[id2]
                neg_id1 += len(df[id1][id2]) * ids_sampled[id2]
        fp += fp_id1 * ids_sampled[id1]
        neg += neg_id1 * ids_sampled[id1]
    fpr_boot = fp / neg
    return (fnr_boot, fpr_boot)


def compute_error_metrics_vertex_boot(args):
    """Computes the fnr and fpr for the vertex bootstrap.

    Parameters:
    args (tuple): A tuple consisting of three arguments:
        df (dict): A dictionary of dictionaries, where the outer dictionary keys are identifiers for different entities, and the inner dictionary keys are the identifiers for the same entities, and the corresponding values are lists of binary labels (1 is error, 0 is correct prediction).
        fpr (float): The false match rate.

    Returns:
        tuple: A tuple of two floats representing the fnr and fpr.
    """
    df, fpr = args
    ids = list(df.keys())
    ids_sampled = Counter(random.choices(ids, k=len(ids)))

    fnr_boot = sum([sum(df[id][id]) * ids_sampled[id] for id in ids_sampled if id in df[id]]) / \
        sum(len(df[id][id]) * ids_sampled[id]
            for id in ids_sampled if id in df[id])
    neg = 0
    fp = 0
    for id1 in ids_sampled:
        fp_id1 = 0
        neg_id1 = 0
        for id2 in df[id1]:
            if id2 in ids_sampled:
                if id2 != id1:
                    fp_id1 += sum(df[id1][id2]) * ids_sampled[id2]
                    neg_id1 += len(df[id1][id2]) * ids_sampled[id2]
                else:
                    fp_id1 += fpr * len(df[id1][id2]) * \
                        (ids_sampled[id2] - 1)
                    neg_id1 += len(df[id1][id2]) * (ids_sampled[id2] - 1)
        fp += fp_id1 * ids_sampled[id1]
        neg += neg_id1 * ids_sampled[id1]
    fpr_boot = fp / neg
    return (fnr_boot, fpr_boot)


def get_ci_boot(df, alpha, B, method="vertex", parallel=False):
    """
    Compute confidence intervals for the false negative rate (fnr) and false
     match rate (fpr) using double-or-nothing bootstrapping.

     Args:
        df (dict): A dictionary where the keys represent the IDs of the individuals being compared,
         and the values are dictionaries with keys representing the IDs of the individuals they are
         being compared to, and the values are lists of 0s and 1s, where 1 indicates an error and 0
         a non error.
        alpha (float): The significance level for the confidence intervals.
        B (int): The number of bootstrap replicates.
        method (character): The method to use for bootstrapping. Either "vertex" or "dbn" for vertex or double-or-nothing bootstrap.
        parallel (bool, optional): whether to parallelize computation. Defaults to False.

     Returns:
         tuple: A tuple of two numpy arrays representing the lower and upper
                bounds of the confidence intervals for the fnr and fpr.
     """

    method_to_error_metrics_fn = {
        "vertex": compute_error_metrics_vertex_boot,
        "dbn": compute_error_metrics_dbn_boot
    }

    compute_error_metrics_boot = method_to_error_metrics_fn.get(method)

    fnr, fpr = compute_error_metrics(df)

    if not compute_error_metrics:
        return -1

    # Compute bootstrapped error metrics in parallel or sequentially
    boot_values = []
    if parallel:
        with mp.Pool() as p:
            boot_values = p.map(compute_error_metrics_boot, [
                                (df, fpr) if method == "vertex" else (df) for _ in range(B)])
    else:
        boot_values = [compute_error_metrics_boot(
            (df, fpr)) if method == "vertex" else compute_error_metrics_boot(df) for _ in range(B)]

    fnr_boot, fpr_boot = [x[0]
                          for x in boot_values], [x[1] for x in boot_values]
    # obtain intervals
    fnr_ci = np.quantile(fnr_boot, q=[math.floor(
        B * alpha / 2) / B, math.ceil(B * (1-alpha / 2))/B])
    fpr_ci = np.quantile(fpr_boot, q=[math.floor(
        B * alpha / 2) / B, math.ceil(B * (1-alpha / 2))/B])

    # if intervals are degenerate at 0, apply correction using Wilson under independence
    # fnr_ci, fpr_ci = correct_boot_cis_by_wilson(df, fnr_ci, fpr_ci, alpha)

    return list(fnr_ci), list(fpr_ci)


def get_ci_wilson(n_errors, n_instances, mean_variance, alpha):
    """
    Calculates the confidence interval for error rate using Wilson score interval method.

    Parameters:
        n_errors (int): Number of errors observed.
        n_instances (int): Total number of instances.
        mean_variance (float): Estimate of the variance of the mean (i.e., of n_errors/n_instances). 
        alpha (float): Significance level, ranging from 0 to 1.

    Returns:
        bounds (list of floats): A list of two floats representing the lower and upper bounds of the confidence interval.
    """
    mu = n_errors / n_instances

    if mu == 0 or mu == 1:
        n_eff = n_instances
    else:
        n_eff = mu * (1 - mu) / mean_variance

    z = stats.norm.ppf(1 - alpha / 2)
    sqrt_term = np.sqrt((mu * (1 - mu) + z**2 / 4 / n_eff) / n_eff)
    bounds = (mu + z**2 / 2 / n_eff + np.array((-1, 1))
              * z * sqrt_term) / (1 + z**2 / n_eff)

    if n_errors == 0:
        bounds[0] = 0
    elif n_errors == n_instances:
        bounds[1] = 1
    elif n_errors == 1:
        bounds[0] = -np.log(1 - alpha) / n_instances
    elif n_errors == n_instances - 1:
        bounds[1] = 1 + np.log(1 - alpha) / n_instances

    return bounds.tolist()


def estimate_var_error_metrics(df):
    """
    Calculates the variance of the error rate via plug-in estimators. 

    Parameters:
        df (dict): A dictionary where the keys represent the IDs of the individuals being compared,
         and the values are dictionaries with keys representing the IDs of the individuals they are
         being compared to, and the values are lists of 0s and 1s, where 1 indicates an error and 0
         a non error.

    Returns:
        variance estimates (list of floats): two floats corresponding to the estimates of the fnr and fpr variances. 
    """
    fnr, fpr = compute_error_metrics(df)

    # estimate fnr variance
    fnr_by_id = [(np.mean(df[id][id]) - fnr) * len(df[id][id])
                 for id in df.keys() if id in df[id] and len(df[id][id]) > 0]
    m_by_id = [len(df[id][id]) for id in df.keys() if id in df[id]]
    var_fnr = np.sum(np.square(fnr_by_id)) / np.square(np.sum(m_by_id))

    # estimate fpr variance
    m_by_id = [[len(df[id1][id2]) for id2 in df[id1] if id2 != id1]
               for id1 in df.keys()]
    fpr_by_id = [[(np.mean(df[id1][id2]) - fpr) * len(df[id1][id2])
                  for id2 in df[id1] if id2 != id1] for id1 in df.keys()]
    var_ids_fpr = sum([sum([y ** 2 for y in x]) for x in fpr_by_id])
    cov_ids_fpr = 0
    for fpr_id1 in fpr_by_id:
        for i, num in np.ndenumerate(fpr_id1):
            rest_of_array = np.delete(fpr_id1, i)
            cov_ids_fpr += num * sum(rest_of_array)

    var_fpr = (2 * var_ids_fpr + 4 * max(cov_ids_fpr, 0)) / \
        (sum([sum(x) for x in m_by_id]) ** 2)

    return var_fnr, var_fpr


# ------------------------------------------------------------------------------
# methods for constructing roc pointwise confidence intervals


def compute_roc_vertex_boot(args):
    df, fpr = args
    ids = list(df.keys())
    ids_sampled = random.choices(ids, k=len(ids))
    ids_sampled = Counter(ids_sampled)

    genuine_scoresb, impostor_scoresb = get_gen_and_imp_scores(
        df, ids_sampled)

    fprb, tprb, thresholds = metrics.roc_curve(y_true=[0] * len(impostor_scoresb) + [
        1] * len(genuine_scoresb), y_score=impostor_scoresb + genuine_scoresb, drop_intermediate=False)
    auc = metrics.auc(fprb, tprb)

    return {'tpr': np.interp(fpr, fprb, tprb).tolist(), 'auc': auc}


def compute_roc_dbn_boot(args):
    df, fpr = args

    ids = list(df.keys())
    weights = random.choices([0, 1], k=len(ids))
    ids_sampled = {ids[i]: 2 for i in range(len(ids)) if weights[i] == 1}

    genuine_scoresb, impostor_scoresb = get_gen_and_imp_scores(
        df, ids_sampled)

    fprb, tprb, thresholds = metrics.roc_curve(y_true=[0] * len(impostor_scoresb) + [
        1] * len(genuine_scoresb), y_score=impostor_scoresb + genuine_scoresb, drop_intermediate=False)
    auc = metrics.auc(fprb, tprb)

    return {'tpr': np.interp(fpr, fprb, tprb).tolist(), 'auc': auc}


def get_ci_roc_boot(df, alpha, B, method="vertex", target_fpr=None, parallel=False):
    """
    Compute pointwise confidence intervals for ROC coordinates (TPR@FPR, FRR@FAR, fnr@fpr) via vertical averaging with vertex bootstrap. 

    Args:
        df (dict): A dictionary where the keys represent the IDs of the individuals being compared, and the values are dictionaries with keys representing the IDs of the individuals they are being compared to, and the values are lists of floats representing the similarity scores between the two individuals.
        alpha (float): significance level for computing confidence intervals.
        B (int): number of bootstrap samples.
        method (character): Either "vertex" or "dbn" for vertex bootstrap or double-or-nothing bootstrap resepectively. Defaults to "vertex".
        target_fpr (list, optional): list of target false positive rates for ROC curve. Defaults to None.
        parallel (bool, optional): whether to parallelize computation. Defaults to False.

    Returns:
        Dict[float, List[float]]: dictionary containing confidence intervals for each FNR@fpr value.
    """

    method_to_error_metrics_fn = {
        "vertex": compute_roc_vertex_boot,
        "dbn": compute_roc_dbn_boot
    }

    compute_roc_boot = method_to_error_metrics_fn.get(method)

    if not compute_roc_boot:
        return -1

    genuine_scores, impostor_scores = get_gen_and_imp_scores(df)
    fpr, tpr, thresholds = metrics.roc_curve(y_true=[0] * len(impostor_scores) + [
                                             1] * len(genuine_scores), y_score=impostor_scores + genuine_scores)
    auc = metrics.auc(fpr, tpr)
    if target_fpr:
        tpr = list(tpr) + \
            list(np.interp(target_fpr, fpr, tpr))
        fpr = list(fpr) + target_fpr
        thresholds = list(thresholds) + [-1] * len(target_fpr)

        # sort by fpr
        tpr, thresholds, fpr = [x for _, x in sorted(zip(fpr, tpr))], [
            x for _, x in sorted(zip(fpr, thresholds))], sorted(fpr)

    if parallel:
        with mp.Pool() as p:
            out_roc = p.map(compute_roc_boot, [
                (df, fpr) for _ in range(B)])
    else:
        out_roc = [compute_roc_boot([df, fpr]) for _ in range(B)]

    confint = np.percentile(
        [x['tpr'] for x in out_roc], q=[alpha/2 * 100, 100 - (alpha/2 * 100)], axis=0)
    confint_auc = np.percentile(
        [x['auc'] for x in out_roc], q=[alpha/2 * 100, 100 - (alpha/2 * 100)], axis=0)

    out = {"fpr": fpr, "tpr": tpr, "tpr_cis": [
        [x[0], x[1]] for x in zip(confint[0], confint[1])], "threshold": thresholds}
    out['auc'], out['auc_cis'] = auc, list(confint_auc)

    # if interval is degenerate at 0, apply correction using Wilson under independence
    for i, tpr_ci in enumerate(out['tpr_cis']):
        if tpr_ci[1] == 0 or tpr_ci[0] == 1:
            pos = len(genuine_scores)
            fn = 0 if tpr_ci[1] == 0 else pos
            out['tpr_cis'][i] = get_ci_wilson(
                n_errors=fn, n_instances=pos, mean_variance=0, alpha=alpha)

    return out


def estimate_threshold_for_fpr(df, target_fpr):
    """
    Estimate the threshold(s) to obtain a given false match rate (fpr).

    Args:
        df (dict): A dictionary where the keys represent the IDs of the individuals being compared, and the values are dictionaries with keys representing the IDs of the individuals they are being compared to, and the values are lists of floats representing the similarity scores between the two individuals.
        target_fpr (float): a float or list of floats representing the target false match rate(s).

    Returns:
        thresholds: a float or list of floats representing the estimated threshold(s) for the given fpr(s).
    """
    dfl = [df[id1][id2] for id1 in df.keys()
           for id2 in df[id1].keys() if id1 != id2]
    dfl = [item for sublist in dfl for item in sublist]
    dfl.sort()
    thresholds = []
    for fpr in target_fpr:
        tnr = 1 - fpr
        thresholds += [dfl[min(math.floor(len(dfl) * tnr), len(dfl) - 1)]]
    return thresholds


def get_ci_roc_wilson_at_threshold(args):
    df, threshold, target_fpr, alpha, alpha_fpr = args

    # make this faster by computing the fpr directly here
    df_error = generate_errors(df, threshold)
    fnr_var, fpr_var = estimate_var_error_metrics(df_error)
    fnr, fpr = compute_error_metrics(df_error)
    neg = sum([len(df[id1][id2]) for id1 in df.keys()
               for id2 in df[id1] if id1 != id2]) / 2
    fpr_ci = get_ci_wilson(
        n_errors=fpr * neg, n_instances=neg, mean_variance=fpr_var, alpha=alpha_fpr)
    thresholds_fpr_bounds = estimate_threshold_for_fpr(df, fpr_ci)

    # iterate over the thresholds for the bounds of the fpr interval
    for j in range(len(fpr_ci)):
        df_error = generate_errors(df, thresholds_fpr_bounds[j])
        fnr_j, fpr = compute_error_metrics(df_error)
        fnr_var, fpr_var = estimate_var_error_metrics(df_error)
        pos = sum([len(df[id][id]) for id in df if id in df[id]])
        fnr_ci = get_ci_wilson(
            n_errors=fnr_j * pos, n_instances=pos, mean_variance=fnr_var, alpha=alpha)
        if j == 0:
            cis = fnr_ci
        else:
            cis[0] = min(cis[0], fnr_ci[0])
            cis[1] = max(cis[1], fnr_ci[1])

    return {"fpr": target_fpr, "tpr": 1 - fnr, "tpr_cis": [1 - x for x in cis[::-1]], "threshold": threshold}


def get_ci_roc_wilson(df, alpha, target_fpr, alpha_fpr=0.1, parallel=False):
    """
    Compute the Wilson Confidence Interval for the False Negative Match Rate (fnr) at a given
    target False Match Rate (fpr) for a set of similarity scores in a dictionary, i.e., pointwise confidence intervals for the ROC curve.

    Args:
        df (dict):A dictionary where the keys represent the IDs of the individuals being compared, and the values are dictionaries with keys representing the IDs of the individuals they are being compared to, and the values are lists of floats representing the similarity scores between the two individuals.
        alpha (float): Significance level for the confidence interval.
        target_fpr (List[float]): List of target False Match Rates for which to estimate the fnr@fpr confidence intervals.
        alpha_fpr (float, optional): Significance level for the fpr intervals. Defaults to 0.1.
        parallel (bool, optional): Whether to use multiprocessing. Defaults to False.

    Returns:
        dict: Dictionary with target fpr as keys and confidence intervals for the fnr as values.
    """

    thresholds = estimate_threshold_for_fpr(df, target_fpr)
    if parallel:
        with mp.Pool() as p:
            out = p.map(get_ci_roc_wilson_at_threshold, [
                (df, thresholds[i], target_fpr[i], alpha, alpha_fpr) for i in range(len(thresholds))])
    else:
        out = []
        for i in range(len(thresholds)):
            out += [get_ci_roc_wilson_at_threshold(
                (df, thresholds[i], target_fpr[i], alpha, alpha_fpr))]

    # combine the results
    return {key: [d[key] for d in out] for key in out[0].keys()}
