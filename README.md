# KG-SLomics: KG-SLomics: Cancer Type-Specific Synthetic Lethality Prediction Using Knowledge Graph and Multiomics Integrated Graph Neural Network

by Songyeon Lee, Hojung Nam

## Abstract
> Synthetic lethality (SL) offers a promising strategy for targeting cancers driven by undruggable mutations. While wet lab experiments for screening gene pairs are cost-intensive, computational approaches, from statistical analysis to deep learning models, have been developed to predict SL pairs. However, these methods often overlook critical challenges: (1) SL interactions can vary significantly between different types of cancer. (2) Most existing approaches incorporate outdated biological networks and gene-specific data as features but fail to account for cancer-specific characteristics. Recent research has begun addressing cancer-specific SL pairs but struggles to generalize findings across multiple cancer types.
We propose KG-SLomics, a graph attention network-based model that generalizes SL prediction using an extensively updated knowledge graph (KG) and multiple cancer cell line data. Our approach first constructs a comprehensive KG with the latest biological entity information, resulting in a three-fold increase in size compared to previous versions. The model harmonizes pre-trained KG embedding vectors with multiomics data to capture both topological and cell line-specific features. Through relational message passing that considers heterogeneous edge types, the model calculates SL probability with high accuracy. In extensive evaluations, KG-SLomics outperformed state-of-the-art baseline models across different experimental settings, including tests on unseen gene pairs and completely unseen genes. Furthermore, our model successfully identified significant biological entities associated with predicted SL pairs and suggested novel therapeutic targets for cancer patients, demonstrating its potential clinical utility.
KG-SLomics is available at https://github.com/GIST-CSBL/KG-SLomics.

## Overview
![Overview_20250123](https://github.com/user-attachments/assets/c2b70dec-dcff-4ff0-a68a-69591d21cfbc)
