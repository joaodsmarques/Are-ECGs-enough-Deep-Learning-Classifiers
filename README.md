# Are ECGs enough? 🫀
This repository presents the original implementation of the paper [Are ECGs enough? Deep learning classification of cardiac anomalies using only electrocardiograms](https://doi.org/10.48550/arXiv.2503.08960) by João D.S. Marques and Arlindo Oliveira.

## Overview 📚
When we face a new problem related to anomaly cardiac detection, there are multiple questions related to the choice of the best networks, the minimum size of a dataset, what results to expect from a setting, what is the best signal length and normalization... In this study, we try to address some of these questions, while demonstrating that transfer learning is very useful in small imbalanced dataset settings. We investigate the performance of multiple neural network architectures in order to assess the impact of various approaches. Moreover, we check whether these practices enhance model generalization when transfer learning is used by pre-training in PTB-XL and CPSC18 and demonstrating our results through a smaller, more challenging dataset for pulmonary embolism (PE) detection. By leveraging transfer learning, we analyze the extent to which we can improve learning efficiency and predictive performance on limited data. 

### Pipeline:

![GitHub Logo](images/pipeline.png)

## Datasets

We use 3 datasets for this research:

![GitHub Logo](images/datasets.png)

## Citation 💬
If you find this work useful, please consider citing our paper:

```bibtex
@misc{AreECGsEnough,
      title={Are ECGs enough? Deep learning classification of cardiac anomalies using only electrocardiograms}, 
      author={Joao D. S. Marques and Arlindo L. Oliveira},
      year={2025},
      eprint={2503.08960},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.08960}, 
}
```



