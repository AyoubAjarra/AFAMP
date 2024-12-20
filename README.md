# Active Fourier Auditor for Estimating Distributional Properties of ML Models

This repository provides implementation of the paper ["Active Fourier Auditor for Estimating Distributional Properties of ML Models"](https://arxiv.org/abs/2410.08111)

## Summary: 

In this paper, we address the problem of auditing the distributional properties of black-box machine learning models. We propose a universal auditing framework that represents these properties using relevant components extracted directly from the model rather than reconstructing it and subsequently employs a plug-in estimator. These components correspond to the model's Fourier coefficients which the auditor estimates within a reduced search space by focusing on large Fourier coefficients.

## Prerequisites:

The prerequesites are given in the file "requirements.txt".

## Citation:

To cite this project, please use:

```bibtex
@article{ajarra2024active,
  title={Active Fourier Auditor for Estimating Distributional Properties of ML Models},
  author={Ajarra, Ayoub and Ghosh, Bishwamittra and Basu, Debabrota},
  journal={arXiv preprint arXiv:2410.08111},
  year={2024}
}
