<h1 align="center">Confidence Intervals for Error Rates in :jigsaw: Matching Tasks</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2306.01198"><img src="https://img.shields.io/badge/paper-arXiv-red" alt="Paper"></a>
    <img src="https://img.shields.io/github/license/awslabs/cis-matching-tasks" alt="Apache-2.0">
</p>

This repository houses the code to construct confidence intervals for
performance metrics in 1:1 matching tasks :jigsaw: such as 1:1 face and speaker
verification. See [our paper](https://arxiv.org/abs/2306.01198) (to appear at
IJCV) for a description of the methods.

<strong>Performance metrics</strong> With this code you can construct $1-\alpha$
confidence intervals $C_{1-\alpha}$ for metrics $\theta^*$ such as

- False Positive Rate (FPR, aka FMR or FAR) and False Negative Rate (FNR, aka
  FNMR or FRR) estimates
- ROC coordinate estimates such as FNR@FPR (aka FNMR@FMR or FRR@FAR)

such that $\mathbb{P}(\theta^*\in C_{\alpha})\geq 1-\alpha$.  

<strong>Methods</strong>. The methods implemented in this repository include
(_parametric_) Wilson intervals, and (_nonparametric_) double-or-nothing,
vertex, subsets, and two-level bootstrap. This repo contains code in both R and
python. Two of the methods that we recommend using (Wilson and double-or-nothing
bootstrap) are implemented in both languages. 

## :rocket: Getting started

In order to intall the package, run 
```
pip install cismat
```

Test your setup using this example that derives confidence intervals for FNMR
and FMR obtained by binarizing the similarity scores at a given threshold:
```
import json
from cismat import MTData, UncertaintyEstimator

# Load embeddings from JSON file into a dictionary structure
df = json.load(open('data/embeddings.json', 'r'))  # Example structure: dictionary[id][image] = embedding
mt = MTData(df)
mt.generate_similarity_scores()  # Generate cosine similarity scores between images

# Set a threshold for determining matches versus non-matches
threshold = 0.9
# Instantiate the class to estimate error rates using similarity scores
uq = UncertaintyEstimator(scores=mt.similarity_scores) # Example structure: dictionary[id1][id2] = [score between image from id1 and id2]
# Compute False Non-Match Rate (FNMR) and False Match Rate (FMR) based on the threshold
fnr, fpr, _ = uq.compute_binerror_metrics(threshold)
fnr, fpr

# Estimate the variance of FNMR and FMR using a plug-in estimator
var_fnr, var_fpr = uq.compute_variance(threshold=threshold, estimator="plugin")
# Calculate 95% Confidence Intervals (CI) for FNMR and FMR using Wilson's method
ci_fnr, ci_fpr = uq.get_binerror_ci(threshold=threshold, var_fnr=var_fnr, var_fpr=var_fpr, alpha=0.05)
ci_fnr, ci_fpr

# Perform double-or-nothing bootstrap estimation to get variance estimates
uq.run_bootstrap(B=1000)  # B is the number of bootstrap samples
# Estimate variance using the bootstrap method
var_fnr_boot, var_fpr_boot = uq.compute_variance(threshold=threshold, estimator="boot")
# Calculate and print 95% CI for FNMR and FMR using bootstrap estimates
ci_fnr_boot, ci_fpr_boot = uq.get_binerror_ci(threshold=threshold, var_fnr=var_fnr_boot, var_fpr=var_fpr_boot, alpha=0.05)
ci_fnr_boot, ci_fpr_boot
```

To generate the intervals more quickly, simply use
```
uq.get_binerror_ci(threshold = threshold, alpha = 0.05)
```
Under the hood, this function runs the variance computations via the plug-in
estimator. 

To obtain pointwise confidence intervals for the ROC with the double-or-nothing
bootstrap, use
```
ci_tpr_at_fnr, ci_auc = uq.get_roc_ci(target_fpr=[0.01, 0.1], alpha = 0.05)
ci_tpr_at_fnr, ci_auc
```

See the code in `examples/morph.ipynb` for a more detailed example on how to use
the package. In case of large datasets, the computations of the uncertainty may
be burdensome. Luckily, the computational speed of the functions in this package
can be substantially improved. Contact me if you are interested in this. 

We have moved all the code related to experiments in the paper to another
branch named `paper`. 

## :books: Citation

To cite our paper/code/package, use

```
@article{fogliato2023confidence,
  title={Confidence Intervals for Error Rates in Matching Tasks: Critical Statistical Analysis and Recommendations},
  author={Fogliato, Riccardo and Patil, Pratik and Perona, Pietro},
  journal={arXiv preprint arXiv:2306.01198},
  year={2023}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more
information.

## License

This project is licensed under the Apache-2.0 License.
