# KG-SLomics: Synthetic Lethality Prediction Using Knowledge Graph and Cancer Type-Specific Multiomics Integrated Graph Neural Network

by Songyeon Lee, Hojung Nam

## Abstract
> Synthetic lethality (SL) is a phenomenon in which the simultaneous alterations of two genes evoke cell death, whereas a mutation of either gene alone does not adversely affect cell survival. After the clinical application of PARP inhibitors, SL has been a promising strategy for the undruggable cancer mutations by targeting their alternative partner genes. While the various statistical and computational methods can predict SL pairs, they often miss key challenges: variation across cancer types and reliance on outdated networks or gene-specific data that ignore cancer-specific features. Recent progress addresses these gaps but struggles to generalize across multiple cancer types. In this paper, we propose KG-SLomics, a relational graph attention network-based model that predicts SL using an extensively updated knowledge graph (KG) and multiple cancer cell line data. We construct a comprehensive KG incorporating newly curated biological entities, tripling its size compared to previous versions. Pre-trained KG embeddings are combined with multiomics data to capture topological and cancer-specific features. Through relational message passing, KG-SLomics calculates SL probabilities with high accuracy, allocating high attention scores to the relevant entities in KG. It outperformed advanced baselines in various evaluations and suggested novel therapeutic targets, underscoring its clinical potential.

## Overview
[Fig1_Overview_renew.tif](https://github.com/user-attachments/files/22586813/Fig1_Overview_renew.tif)

## Requirements
numpy                     1.24.3 <br/>
pandas                    2.2.3 <br/>
torch                     2.6.0 <br/>
torch-cluster             1.6.3+pt21cu118 <br/>
torch-geometric           2.6.1 <br/>
torch-scatter             2.1.2+pt21cu118 <br/>
torch-sparse              0.6.18+pt21cu118 <br/>
torch-spline-conv         1.2.2+pt21cu118 <br/>
scikit-learn              1.6.1 <br/>
scipy                     1.15.2 <br/>
