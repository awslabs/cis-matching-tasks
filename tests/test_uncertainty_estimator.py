import unittest
import json
from cismat import MTData, UncertaintyEstimator
import numpy as np

class TestUncertaintyEstimator(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        with open('data/embeddings.json', 'r') as file:
            df = json.load(file)
        cls.mt = MTData(df)
        cls.mt.generate_similarity_scores()
    
    def test_uncertainty_estimator(self):
        uq = UncertaintyEstimator(scores=self.mt.similarity_scores)
        threshold = 0.9
        uq.run_bootstrap(B=1000)

        fnr, fpr, _ = uq.compute_binerror_metrics(threshold)
        self.assertGreaterEqual(fnr, 0.0)
        self.assertLessEqual(fnr, 1.0)
        self.assertGreaterEqual(fpr, 0.0)
        self.assertLessEqual(fpr, 1.0)
        
        var_fnr, var_fpr = uq.compute_variance(threshold, estimator="plugin")
        var_fnr_boot, var_fpr_boot = uq.compute_variance(threshold, estimator="boot")
        auc = uq.compute_auc()
        ci_bin_error_plugin = uq.get_binerror_ci(threshold, var_fnr, var_fpr, alpha=0.05)
        ci_bin_error_boot = uq.get_binerror_ci(threshold, var_fnr_boot, var_fpr_boot, alpha=0.05)

        roc_cis, auc_cis = uq.get_roc_ci(target_fpr=[0.1, 0.5], alpha=0.05)

        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)

if __name__ == '__main__':
    unittest.main()
