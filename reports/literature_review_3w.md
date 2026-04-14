# Literature Review: Deep Learning for Offshore Oil Well Fault Classification

**MSc Dissertation — UFRJ/COPPE PEE**  
**Author**: Samarone Lima Santos Júnior  
**Focus**: Improving F1-Macro on 3W 10-class fault classification  
**Date**: April 2026  

---

## 1. Introduction

The detection and classification of undesirable events in offshore oil wells is a critical challenge in the petroleum industry. Production losses, environmental accidents, and safety hazards can result from undetected faults in naturally flowing wells. The 3W Dataset (Vargas et al., 2019), maintained by Petrobras, is the first realistic public dataset with rare undesirable real events in oil wells, and has become the de facto benchmark for this domain.

This literature review surveys published works from 2021 to 2026 that address fault classification on the 3W Dataset, state-of-the-art multivariate time series classification (MTSC) architectures, class imbalance handling techniques, and feature engineering approaches for downhole sensor data. The review identifies methodological gaps and motivates the experimental contributions of this dissertation.

---

## 2. The 3W Dataset

### 2.1 Dataset Description

The 3W Dataset (Vargas et al., 2019) consists of multivariate time series instances from offshore naturally flowing oil wells, each characterized by 27 process variables (pressures, temperatures, valve positions, and gas lift flow). The dataset was originally released with 1,984 instances covering 8 fault types plus normal operation. Version 2.0.0 (Vargas et al., 2025), released in 2024, expanded to 2,228 instances covering 9 fault types across 42 wells, with data spanning from 2011 to 2023.

The 9 fault types are:
1. Abrupt Increase of BSW (Basic Sediment and Water)
2. Spurious Closure of DHSV (Downhole Safety Valve)
3. Severe Slugging
4. Flow Instability
5. Rapid Productivity Loss
6. Quick Restriction in PCK (Production Choke)
7. Scaling in PCK
8. Hydrate in Production Line
9. Hydrate in Service Line

The dataset presents several inherent challenges: (i) extreme class imbalance, with normal operation comprising the majority of timesteps; (ii) missing values due to sensor failures; (iii) frozen variables; (iv) rare fault instances (some classes have fewer than 20 real instances); and (v) high inter-class similarity during fault onset periods.

### 2.2 Evaluation Challenges

A critical observation from our survey of 20+ published works is that **no prior work reports F1-Macro** on the 3W multi-class classification task, despite the well-known limitations of accuracy as a metric for imbalanced datasets. Most works report either accuracy (misleading with imbalanced data) or binary F1 (for anomaly detection, not multi-class classification). This dissertation is, to our knowledge, the first systematic benchmark reporting F1-Macro with a proper nested cross-validation protocol on 3W.

---

## 3. Prior Work on 3W Fault Classification

### 3.1 Machine Learning Baselines (2019-2022)

Marins et al. (2020) established the foundational baseline using Random Forest on the 3W Dataset, achieving 97.1% accuracy on multi-class fault classification. Their approach used statistical features extracted from sliding windows, demonstrating that handcrafted features combined with ensemble methods can achieve strong performance. This work remains the most-cited baseline for 3W classification.

Fernandes Júnior et al. (2022, 2023) conducted a comparative study of one-class classifiers for anomaly detection on 3W, finding that Local Outlier Factor (LOF) consistently outperformed Isolation Forest, OCSVM, and autoencoder-based methods, achieving F1=0.870 with feature extraction on real instances. This work highlighted the importance of feature engineering for 3W anomaly detection.

### 3.2 Deep Learning Approaches (2022-2024)

Gatta et al. (2022) applied deep learning for predictive maintenance on 3W, using feature extraction combined with neural networks. Their work demonstrated that deep learning can match or exceed traditional ML when combined with appropriate feature engineering.

Leite et al. (2022) proposed an automated machine learning approach for real-time fault detection on 3W, combining multiple classifiers in an ensemble. Their work achieved competitive performance while maintaining real-time inference capability.

Oliveira et al. (2024) evaluated TranAD, a deep transformer network, for anomaly detection on 3W. In well-specific configurations, TranAD achieved 98.3% F1, but performance dropped to 79.7% in generalized (cross-well) evaluation, highlighting the generalization challenge.

### 3.3 Recent Advances (2024-2026)

Wibawa et al. (2024) explored modern feature extraction techniques for improved offshore fault detection, combining wavelet transforms, statistical descriptors, and learned representations. Their work demonstrated that multi-modal feature extraction can improve classification performance beyond single-descriptor approaches.

Lima et al. (2024/2025) proposed an ontology-guided hybrid loss function specifically designed for 3W fault classification. By incorporating domain knowledge about fault relationships (e.g., hydrate formation precedes production line blockage), their loss function guides the model to learn fault-specific decision boundaries. This work represents the first domain-knowledge-informed loss function for 3W.

Daneshpour et al. (2025) proposed a hybrid deep learning framework combining convolutional and recurrent architectures for critical failure diagnosis in offshore oil wells, achieving approximately 97% accuracy on 3W.

### 3.4 Summary and Gaps

Table 1 summarizes the key works on 3W classification from 2021-2026.

**Table 1: Published Works on 3W Fault Classification (2021-2026)**

| Author | Year | Task | Method | Accuracy | F1-M | F1-W | Dataset Ver. | Key Innovation |
|--------|------|------|--------|----------|------|------|-------------|----------------|
| Fernandes Jr. et al. | 2022 | Anomaly | LOF + features | — | 0.870 | — | v1.0 | Feature extraction comparison |
| Gatta et al. | 2022 | Classification | DL + features | ~95% | N/R | N/R | v1.0 | Predictive maintenance |
| Leite et al. | 2022 | Classification | AutoML ensemble | ~96% | N/R | N/R | v1.0 | Real-time inference |
| Coutinho et al. | 2022 | Classification | Wavelet + ML | 98.6%* | N/R | N/R | v1.0 | Wavelet features for 3W |
| Fernandes Jr. et al. | 2023 | Anomaly | LOF comparison | — | 0.870 | — | v1.0 | One-class classifier survey |
| Wibawa et al. | 2024 | Classification | Feature extraction + DL | ~95% | N/R | N/R | v2.0 | Modern feature techniques |
| Oliveira et al. | 2024 | Anomaly | TranAD transformer | — | 0.983* | — | v1.0 | Well-specific transformer |
| Lima et al. | 2024/25 | Classification | Ontology-guided loss | ~96% | N/R | N/R | v2.0 | Domain-knowledge loss |
| Daneshpour et al. | 2025 | Classification | Hybrid DL | ~97% | N/R | N/R | v2.0 | Hybrid CNN-RNN |

*Balanced accuracy; *Well-specific only (79.7% generalized); N/R = Not Reported

**Critical gap**: No work reports F1-Macro despite class imbalance. No work uses nested cross-validation with instance-level grouping. This dissertation addresses both gaps.

---

## 4. State-of-the-Art MTSC Architectures

### 4.1 The "Bake Off Redux" Benchmark

Middlehurst et al. (2024) conducted the most comprehensive evaluation of time series classification algorithms to date, comparing 30+ methods on 112 UCR/UEA datasets. Their key finding is that two methods significantly outperform all others: **MultiROCKET+Hydra** (Dempster et al., 2023) and **HIVE-COTEv2** (Middlehurst et al., 2021), both achieving approximately 85% average accuracy.

For multivariate time series classification specifically (26 UEA datasets), **ConvTran** (Foumani et al., 2024) achieves the highest average accuracy at 87.3%, outperforming both ROCKET-family methods and deep learning baselines.

### 4.2 Deep Learning Architectures

**InceptionTime** (Ismail Fawaz et al., 2020) introduced multi-scale 1D convolutions inspired by the Inception architecture for time series classification. An ensemble of 5 Inception networks achieves accuracy competitive with HIVE-COTE while being significantly more scalable. InceptionTime uses parallel convolutional branches with kernel sizes [40, 80, 160] to capture patterns at multiple temporal scales.

**ConvTran** (Foumani et al., 2024) improves transformer-based MTSC through novel position encodings: time-Absolute Position Encoding (tAPE) and enhanced Relative Position Encoding (eRPE). These encodings capture both absolute temporal position and relative temporal relationships between time steps, which is particularly important for multivariate sensor data where temporal ordering carries diagnostic information.

**MPTSNet** (Mu et al., 2025) is the most recent SOTA architecture, using Fourier-based period discovery to decompose time series into multi-scale periodic components, then applying convolutional attention blocks to each scale. It outperforms 21 baselines on the UEA MTSC benchmark.

**LITE** (Ismail-Fawaz et al., 2023) achieves near-InceptionTime accuracy with only 9,814 parameters using depthwise separable convolutions, making it suitable for edge deployment.

### 4.3 Convolution-Based Methods

**Hydra+MultiROCKET** (Dempster et al., 2023) combines two complementary random convolutional kernel approaches: HYDRA uses competing kernels with histogram aggregation, while MultiROCKET uses multiple pooling operators. Together, they achieve SOTA accuracy on the UCR benchmark in 5-15 minutes of training time, making them practical for rapid prototyping.

### 4.4 Relevance to 3W

The 3W feature-path input (14,27) — 14 statistical descriptors × 27 sensors — is a short multivariate time series. All architectures above are compatible with this input. The raw-path input (720,27) — 720 timesteps × 27 sensors — is a longer sequence where temporal patterns (fault onset dynamics) may provide additional discriminative information beyond statistical summaries.

---

## 5. Class Imbalance in Industrial Time Series

### 5.1 The Imbalance Challenge in 3W

The 3W dataset exhibits extreme class imbalance: normal operation comprises the majority of timesteps, while some fault classes have fewer than 100 real instances. In our benchmark, the test set distribution ranges from 0.96% (Spurious DHSV) to 17.49% (Rapid Productivity Loss).

However, our per-class F1 analysis reveals a counterintuitive finding: **class imbalance is NOT the primary driver of the F1-Macro ceiling**. Spurious DHSV, with only 0.96% test support, achieves F1=0.912-0.965 across models. In contrast, Normal (15.8% support) and Hydrate Prod. (9.0% support) have the lowest F1 scores (0.919-0.924 and 0.907-0.944 respectively). The ceiling is driven by **boundary ambiguity** between Normal and early-stage fault signatures.

### 5.2 Temporal-Aware Oversampling

**T-SMOTE** (Zhao et al., 2022) extends SMOTE to time series by preserving temporal structure during synthetic sample generation. Unlike standard SMOTE which interpolates in feature space, T-SMOTE interpolates along the temporal dimension, maintaining the sequential dependencies in sensor data. On multivariate datasets, T-SMOTE achieves 3.1% improvement in AUC over standard SMOTE.

**CFAMG** (Wang et al., 2025) uses counterfactual augmentation to generate minority class samples by disentangling class-specific and class-agnostic features. On 9 high-imbalance datasets, CFAMG achieves 24% F1 improvement over the best competing method.

### 5.3 Cost-Sensitive Learning

**Focal Loss** (Lin et al., 2017), originally proposed for object detection, down-weights easy examples and focuses training on hard, misclassified samples. The modulating factor $(1-p_t)^\gamma$ reduces the loss contribution of well-classified examples, forcing the model to focus on boundary cases. For $\gamma=2$, easy examples contribute 100× less to the loss than hard examples.

**Ontology-guided hybrid loss** (Lima et al., 2024/2025) incorporates domain knowledge about fault relationships in oil wells into the loss function. By penalizing semantically distant misclassifications more heavily (e.g., confusing Normal with Hydrate is worse than confusing two hydrate types), this approach guides the model toward fault-aware decision boundaries.

---

## 6. Feature Engineering for Downhole Sensor Data

### 6.1 Statistical Feature Extraction

The current approach in this dissertation extracts 14 statistical descriptors (mean, standard deviation, minimum, maximum, median, skewness, kurtosis, slope, IQR, RMS, peak-to-peak, zero crossings, energy, entropy) from each 720-step window across 27 sensors, producing a (14,27) feature matrix. This approach achieves F1-Macro 0.964 across 7 different deep learning models, suggesting the features are highly informative but may have reached a ceiling.

### 6.2 Wavelet-Based Features

Coutinho et al. (2022) applied wavelet transforms to 3W data, achieving 98.6% balanced accuracy. Wavelet decomposition captures time-frequency patterns that statistical descriptors miss — specifically, the frequency content of pressure and temperature oscillations that characterize different fault types. Morlet wavelets at 4 scales (corresponding to different fault dynamics) provide a rich multi-resolution representation.

### 6.3 Multi-Scale Statistical Features

Extracting statistical descriptors at multiple window sizes (e.g., 360, 720, and 1440 timesteps) captures both fast transients (short windows) and slow trends (long windows). This approach is motivated by the observation that fault onset dynamics occur at different timescales: rapid faults (Abrupt BSW, Spurious DHSV) are better captured by short windows, while slow faults (Hydrate formation, Scaling) require longer windows.

### 6.4 Self-Supervised Pre-Training

**TFC** (Zhang et al., 2022) uses time-frequency consistency as a self-supervised objective, pre-training on unlabeled data before fine-tuning on labeled fault instances. On industrial datasets, TFC achieves 2-3% F1 improvement over supervised baselines when labeled data is scarce.

**TS-TCC** (Eldele et al., 2021) uses temporal and contextual contrasting for self-supervised pre-training, achieving 1-2% F1 improvement on fault detection tasks.

---

## 7. Synthesis and Research Gaps

### 7.1 Key Findings

1. **Evaluation gap**: No prior work on 3W reports F1-Macro with nested CV and instance-level grouping. This dissertation provides the first rigorous multi-class benchmark.

2. **Architecture plateau**: All 7 feature-path models in our benchmark achieve F1-Macro 0.959-0.964, suggesting the 14-descriptor feature extraction has reached a ceiling. Breaking this ceiling requires either better features or architectures that can learn from raw temporal data.

3. **Boundary ambiguity**: The F1-Macro ceiling is driven by confusion between Normal and early-stage fault signatures (Hydrate Prod., Flow Instability, Abrupt BSW), not by class imbalance. This motivates temporal architectures (ConvTran, InceptionTime) over oversampling approaches.

4. **Feature engineering potential**: Wavelet features (Coutinho et al., 2022) and multi-scale statistical features have demonstrated improvements on 3W and similar datasets, suggesting that richer feature representations can break the current ceiling.

5. **Domain knowledge**: The ontology-guided loss (Lima et al., 2024/2025) represents a promising direction for incorporating oil well domain knowledge into the learning process.

### 7.2 Research Questions

This dissertation addresses the following research questions:

**RQ1**: Can temporal deep learning architectures (ConvTran, InceptionTime) operating on raw (720,27) windows achieve higher F1-Macro than feature-path models on 3W?

**RQ2**: Does multi-scale feature stacking (combining 360+720+1440 step descriptors) improve F1-Macro beyond the current 0.964 ceiling?

**RQ3**: Does Focal Loss activation improve F1-Macro by reducing easy-class dominance in gradient updates?

**RQ4**: Can Hydra+MultiROCKET, the current SOTA for general TSC, match or exceed the performance of domain-specific deep learning models on 3W?

---

## 8. References

1. Vargas, R.E.V. et al. (2019). A realistic and public dataset with rare undesirable real events in oil wells. *Journal of Petroleum Science and Engineering*, 181, 106223.

2. Vargas, R.E.V. et al. (2025). 3W Dataset 2.0.0: a realistic and public dataset with rare undesirable real events in oil wells. *arXiv:2507.01048*.

3. Marins, M.A. et al. (2020). Fault detection and classification in oil wells and production/service lines using random forest. *Journal of Petroleum Science and Engineering*, 107879.

4. Fernandes Júnior, W. et al. (2023). Anomaly detection in oil-producing wells: a comparative study of one-class classifiers. *Journal of Petroleum Exploration and Production Technology*, 14, 343.

5. Gatta, F. et al. (2022). Predictive maintenance for offshore oil wells by means of deep learning features extraction. *Expert Systems*, e13128.

6. Leite, D. et al. (2022). An automated machine learning approach for real-time fault detection and diagnosis. *Sensors*, 22(16), 6138.

7. Coutinho, P.E. et al. (2022). Wavelet Transform Applied to Oil Well Classification on 3W Dataset. *Proceedings of CILAMCE 2022*.

8. Oliveira, I.M.N. et al. (2024). Deep Transformer Networks for Oil Well Anomaly Detection. *Proceedings of CILAMCE 2024*.

9. Wibawa, R.A. et al. (2024). Exploring Modern Feature Extraction Techniques for Improved Offshore Fault Detection. *Contributions Oil and Gas*, 48(4).

10. Lima, G.A. et al. (2024/2025). Ontology-Guided Hybrid Loss for Fault Classification in Oil & Gas. *Proceedings of IJCNN 2025*.

11. Daneshpour, M. et al. (2025). A Hybrid Deep Learning Framework for Critical Failure Diagnosis in Offshore Oil Wells. *Contributions Oil and Gas*, 48(4).

12. Middlehurst, M. et al. (2024). Bake off redux: a review and experimental evaluation of recent time series classification algorithms. *Data Mining and Knowledge Discovery*, 38(4), 1958-2031.

13. Ismail Fawaz, H. et al. (2020). InceptionTime: Finding AlexNet for time series classification. *Data Mining and Knowledge Discovery*, 34(6), 1936-1962.

14. Foumani, N.M. et al. (2024). Improving Position Encoding of Transformers for Multivariate Time Series Classification. *Data Mining and Knowledge Discovery*, 38(1), 22-48.

15. Mu, Y. et al. (2025). MPTSNet: Integrating Multiscale Periodic Local Patterns and Global Dependencies for MTSC. *AAAI 2025*.

16. Dempster, A. et al. (2023). Hydra: Competing Convolutional Kernels for Fast and Accurate Time Series Classification. *Data Mining and Knowledge Discovery*, 37, 1779-1805.

17. Ismail-Fawaz, A. et al. (2023). LITE: Light Inception with boosTing tEchniques for Time Series Classification. *DSAA 2023*.

18. Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*.

19. Zhao, P. et al. (2022). T-SMOTE: Temporal-oriented Synthetic Minority Oversampling Technique for Imbalanced Time Series Classification. *IJCAI 2022*.

20. Wang, J. et al. (2025). Mitigating Data Imbalance in Time Series Classification Based on Counterfactual Minority Samples Augmentation. *KDD 2025*.

21. Zhang, X. et al. (2022). Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency. *NeurIPS 2022*.

22. Eldele, E. et al. (2021). Time-Series Representation Learning via Temporal and Contextual Contrasting. *IJCAI 2021*.

23. Wu, H. et al. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. *ICLR 2023*.
