import numpy as np
import itertools
import random
from sklearn import metrics
from scipy import stats
import tqdm


class UncertaintyEstimator:

    def __init__(self, scores=None):
        self.scores = scores
        self.boot_results = None

    def gather_scores(self, ids_sampled=None):
        df = self.scores
        if ids_sampled is not None:
            df = {
                id1: {id2: df[id1][id2] for id2 in df[id1] if id2 in ids_sampled}
                for id1 in ids_sampled
                if id1 in df
            }

        scores, labels = [], []
        for id1, inner_dict in df.items():
            for id2, values in inner_dict.items():
                if id1 == id2:
                    scores.extend(values)
                    labels.extend([1] * len(values))
                else:
                    scores.extend(values)
                    labels.extend([0] * len(values))
        return scores, labels

    def _run_bootstrap_dbn(self):
        ids_sampled = set(
            [id for id in self.scores.keys() if np.random.choice([0, 1]) == 1]
        )

        scores, labels = self.gather_scores(ids_sampled)
        fprb, tprb, thresholds = metrics.roc_curve(
            y_true=labels, y_score=scores, drop_intermediate=False
        )

        # genuine_scoresb, impostor_scoresb = self._get_gen_and_imp_scores(ids_sampled = ids_sampled)

        # fprb, tprb, thresholds = metrics.roc_curve(y_true=[0] * len(impostor_scoresb) + [
        #     1] * len(genuine_scoresb), y_score=impostor_scoresb + genuine_scoresb, drop_intermediate=False)

        return fprb, tprb, thresholds

    def compute_binerror_metrics(self, threshold):

        scores, labels = self.gather_scores()
        tn, fp, fn, tp = metrics.confusion_matrix(
            y_true=labels, y_pred=np.array(scores) > threshold
        ).ravel()
        self.tn, self.fp, self.fn, self.tp = tn, fp, fn, tp
        fnr, fpr, ppv = fn / (tp + fn), fp / (tn + fp), tp / (fp + tp)

        return fnr, fpr, ppv

    def compute_auc(self):

        scores, labels = self.gather_scores()

        return metrics.roc_auc_score(y_true=labels, y_score=scores)

    # def _get_gen_and_imp_scores(self, ids_sampled=None):

    #     if ids_sampled is None:
    #         ids_sampled = set(self.scores.keys())

    #     filtered_scores = {id1: {id2: self.scores[id1][id2] for id2 in self.scores[id1] if id2 in ids_sampled} for id1 in ids_sampled if id1 in self.scores}
    #     impostor_scores = [filtered_scores[id1][id2] for id1 in filtered_scores for id2 in filtered_scores[id1] if id1 != id2]
    #     genuine_scores = [filtered_scores[id][id] for id in filtered_scores if id in filtered_scores[id]]

    #     return list(itertools.chain.from_iterable(genuine_scores)), list(itertools.chain.from_iterable(impostor_scores))

    def run_bootstrap(self, B=100):
        # TODO: add vertex bootstrap
        boot_results = [self._run_bootstrap_dbn() for B in tqdm.tqdm(range(B))]
        self.boot_results = boot_results

    def _generate_data_errors(self, threshold):
        df_error = {id: {} for id in self.scores.keys()}
        for id1 in self.scores.keys():
            for id2 in self.scores[id1].keys():
                values = self.scores[id1][id2]
                df_error[id1][id2] = [
                    (
                        1
                        if (v < threshold and id1 == id2)
                        or (v > threshold and id1 != id2)
                        else 0
                    )
                    for v in values
                ]
        return df_error

    def _estimate_plugin(self, threshold):

        # genuine_scoresb, impostor_scoresb = self._get_gen_and_imp_scores()

        # fprb, tprb, thresholds = metrics.roc_curve(y_true=[0] * len(impostor_scoresb) + [
        #     1] * len(genuine_scoresb), y_score=impostor_scoresb + genuine_scoresb, drop_intermediate=False)

        fnr, fpr, _ = self.compute_binerror_metrics(threshold)

        # estimate fnr variance
        fnr_by_id = [
            (np.mean(self.errors[id][id]) - fnr) * len(self.errors[id][id])
            for id in self.errors.keys()
            if id in self.errors[id] and len(self.errors[id][id]) > 0
        ]
        m_by_id = [
            len(self.errors[id][id])
            for id in self.errors.keys()
            if id in self.errors[id]
        ]
        var_fnr = np.sum(np.square(fnr_by_id)) / np.square(np.sum(m_by_id))

        # estimate fpr variance
        m_by_id = [
            [len(self.errors[id1][id2]) for id2 in self.errors[id1] if id2 != id1]
            for id1 in self.errors.keys()
        ]
        fpr_by_id = [
            [
                (np.mean(self.errors[id1][id2]) - fpr) * len(self.errors[id1][id2])
                for id2 in self.errors[id1]
                if id2 != id1
            ]
            for id1 in self.errors.keys()
        ]
        var_ids_fpr = sum([sum([y**2 for y in x]) for x in fpr_by_id])
        cov_ids_fpr = 0
        for fpr_id1 in fpr_by_id:
            for i, num in np.ndenumerate(fpr_id1):
                rest_of_array = np.delete(fpr_id1, i)
                cov_ids_fpr += num * sum(rest_of_array)

        var_fpr = (2 * var_ids_fpr + 4 * max(cov_ids_fpr, 0)) / (
            sum([sum(x) for x in m_by_id]) ** 2
        )

        return var_fnr, var_fpr

    def compute_variance(self, threshold, estimator="plugin"):

        if estimator == "plugin":
            self.errors = self._generate_data_errors(threshold)
            var_fnr, var_fpr = self._estimate_plugin(threshold)

        elif estimator == "boot":
            if self.boot_results is None:
                ValueError("Bootstrap needs to be run first")
            indices_threshold = [
                np.argmin(np.abs(boot_th[2] - threshold))
                for boot_th in self.boot_results
            ]
            var_fpr = np.var(
                [
                    self.boot_results[i][0][indices_threshold[i]]
                    for i in range(len(indices_threshold))
                ]
            )
            var_fnr = np.var(
                [
                    self.boot_results[i][1][indices_threshold[i]]
                    for i in range(len(indices_threshold))
                ]
            )

        return var_fnr, var_fpr

    def get_binerror_ci(self, threshold=None, var_fnr=None, var_fpr=None, alpha=0.05):

        if var_fnr is None or var_fpr is None:
            if threshold is None:
                ValueError("A threshold must be specified!")
            var_fnr, var_fpr = self.compute_variance(
                threshold=threshold, estimator="plugin"
            )

        fpr_interval = self._get_ci_wilson(
            n_errors=self.fp,
            n_instances=self.fp + self.tn,
            mean_variance=var_fpr,
            alpha=alpha,
        )
        fnr_interval = self._get_ci_wilson(
            n_errors=self.fn,
            n_instances=self.fn + self.tp,
            mean_variance=var_fnr,
            alpha=alpha,
        )

        return fnr_interval, fpr_interval

    def get_roc_ci(self, target_fpr=None, alpha=0.05):

        if self.boot_results is None:
            ValueError("Bootstrap needs to be run first")

        scores, labels = self.gather_scores()
        auc = self.compute_auc()
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=labels,
            y_score=scores,
            drop_intermediate=False if target_fpr is not None else True,
        )

        if target_fpr is not None:
            tpr = list(tpr) + list(np.interp(target_fpr, fpr, tpr))
            fpr = list(fpr) + target_fpr
            thresholds = list(thresholds) + [-1] * len(target_fpr)

        # sort by fpr
        tpr, thresholds, fpr = (
            [x for _, x in sorted(zip(fpr, tpr))],
            [x for _, x in sorted(zip(fpr, thresholds))],
            sorted(fpr),
        )

        auc_boot = [metrics.auc(fprb, tprb) for fprb, tprb, _ in self.boot_results]
        confint_auc = np.percentile(
            auc_boot, q=[alpha / 2 * 100, 100 - (alpha / 2 * 100)], axis=0
        )

        if target_fpr is not None:
            tpr_at_fpr_boot = [
                np.interp(target_fpr, fprb, tprb).tolist()
                for fprb, tprb, _ in self.boot_results
            ]
        else:
            tpr_at_fpr_boot = [
                np.interp(fpr, fprb, tprb).tolist()
                for fprb, tprb, _ in self.boot_results
            ]
        confint = np.percentile(
            tpr_at_fpr_boot, q=[alpha / 2 * 100, 100 - (alpha / 2 * 100)], axis=0
        )
        return confint, confint_auc

    def get_roc(self, target_fpr=None):

        scores, labels = self.gather_scores()
        auc = self.compute_auc()
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=labels,
            y_score=scores,
            drop_intermediate=False if target_fpr is not None else True,
        )
        auc = metrics.auc(fpr, tpr)

        if target_fpr is None:
            return fpr, tpr, auc

        return fpr, np.interp(target_fpr, fpr, tpr), auc

    def _get_ci_wilson(self, n_errors, n_instances, mean_variance, alpha):
        mu = n_errors / n_instances

        if mu == 0 or mu == 1:
            n_eff = n_instances
        else:
            n_eff = mu * (1 - mu) / mean_variance

        z = stats.norm.ppf(1 - alpha / 2)
        sqrt_term = np.sqrt((mu * (1 - mu) + z**2 / 4 / n_eff) / n_eff)
        bounds = (mu + z**2 / 2 / n_eff + np.array((-1, 1)) * z * sqrt_term) / (
            1 + z**2 / n_eff
        )

        if n_errors == 0:
            bounds[0] = 0
        elif n_errors == n_instances:
            bounds[1] = 1
        elif n_errors == 1:
            bounds[0] = -np.log(1 - alpha) / n_instances
        elif n_errors == n_instances - 1:
            bounds[1] = 1 + np.log(1 - alpha) / n_instances

        return bounds.tolist()
