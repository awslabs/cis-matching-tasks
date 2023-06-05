<h1 align="center">Confidence Intervals for Error Rates in Matching Tasks</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2306.01198"><img src="https://img.shields.io/badge/paper-arXiv-red" alt="Paper"></a>
    <img src="https://img.shields.io/github/license/awslabs/cis-matching-tasks" alt="Apache-2.0">
</p>

This repository houses the code to construct confidence intervals for performance metrics in tasks such as 1:1 face and speaker
verification. See [our paper](https://arxiv.org/abs/2306.01198) for a description of the methods.

<strong>Performance metrics</strong>. The code enables you to construct $1-\alpha$ confidence intervals for

- False Positive Rate (FPR, aka FMR or FAR) and False Negative Rate (FNR, aka FNMR or FRR) estimates
- ROC coordinate estimates such as FNR@FPR (aka FNMR@FMR or FRR@FAR)

<strong>Methods</strong>. The methods implemented in this repository include (_parametric methods_) Wilson intervals, and (_nonparametric methods_) double-or-nothing, vertex, subsets, and two-level bootstrap.

## :computer: Code

The implementation encompasses both R and python languages. The construction of intervals in `R` is situated in [`R-code/utils.R`](R-code/utils.R) and [`R-code/utils-roc.R`](R-code/utils_roc.R). Most of the methods are also accessible in `python` via [`python-code/utils.py`](python-code/utils.py). A Jupyter notebook illustrating the usage of these methods is located at [`python-code/morph_tutorial.ipynb`](python-code/morph_tutorial.ipynb).

Code to replicate the experiments can be found in `R-code/run_experiments_*.R`.

## :books: Citation

To cite our code,

```
@article{fogliato2023confidence,
    title={Confidence Intervals for Error Rates in Matching Tasks: Critical Review and Recommendations},
    author={Riccardo Fogliato and Pratik Patil and Pietro Perona},
    journal={arXiv preprint arXiv:2306.01198},
    year={2023}
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
