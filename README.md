# Health Metrics Clustering Optimization Project

## Master's of Science in Public Health Data Science
## Advanced Machine Learning - Unit Project

---

## Project Overview

This project presents a comprehensive analysis of unsupervised machine learning techniques for health metrics clustering in metabolic phenotype discovery. Through systematic hyperparameter tuning across 19 health metrics and 4 clustering algorithms, we identify optimal configurations for different analytical use cases.

### Key Objectives

1. **Evaluate** multiple clustering algorithms for health data analysis
2. **Optimize** hyperparameters through systematic grid search
3. **Compare** algorithm performance across different health metrics
4. **Analyze** trade-offs between statistical performance and clinical utility
5. **Achieve** target Silhouette Score of 0.87+

### Key Achievements

- **Maximum Silhouette Score**: 0.7866 (Triglycerides + Single Linkage)
- **Algorithm Ranking**: Single Linkage > Average Linkage > Complete Linkage > K-Means
- **Top Performing Metrics**: Triglycerides, Vitamin D, Systolic BP, WBC, Creatinine

---

## Project Structure

```
health-metrics-clustering/
├── notebooks/
│   └── health_metrics_clustering_analysis.ipynb
├── docs/
│   ├── PROJECT_DOCUMENTATION.md
│   ├── FINAL_RESULTS.csv
│   ├── clustering_results.csv
│   ├── distributions_plot.png
│   ├── algorithm_comparison.png
│   ├── detailed_metric_analysis.png
│   ├── hyperparameter_tuning.png
│   └── final_summary.png
├── src/
│   ├── run_comprehensive_metrics_analysis.py
│   ├── run_target_achieved.py
│   └── run_max_optimization.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Installation

```bash
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run comprehensive metric analysis
python src/run_comprehensive_metrics_analysis.py

# Run maximum optimization
python src/run_max_optimization.py
```

### View Results

Open the Jupyter notebook for interactive exploration:

```bash
jupyter notebook notebooks/health_metrics_clustering_analysis.ipynb
```

---

## Methodology Summary

### Dataset

- **Source**: NHANES (National Health and Nutrition Examination Survey)
- **Size**: 5,000 samples, 48 features
- **Metrics Analyzed**: 19 numerical health metrics

### Clustering Algorithms

1. **Single Linkage Hierarchical**: Best for anomaly detection
2. **Complete Linkage Hierarchical**: Best for compact clusters
3. **Average Linkage Hierarchical**: Balanced approach
4. **K-Means**: Industry standard baseline

### Evaluation Metrics

- **Silhouette Score**: Primary metric (-1 to 1, higher is better)
- **Calinski-Harabasz Index**: Variance ratio criterion
- **Davies-Bouldin Index**: Similarity measure (lower is better)

### Hyperparameter Tuning

- **Cluster Counts**: [2, 3, 4, 5]
- **Total Configurations**: 304 (19 metrics × 4 algorithms × 4 cluster counts)

---

## Results Summary

### Best Configurations

| Rank | Metric | Algorithm | Silhouette | Cluster Sizes |
|------|--------|-----------|------------|---------------|
| 1 | Triglycerides | Single Linkage | 0.7866 | [4989, 11] |
| 2 | Triglycerides | Average Linkage | 0.7424 | [4891, 109] |
| 3 | Triglycerides | Complete Linkage | 0.6969 | [561, 4439] |
| 4 | Triglycerides | K-Means | 0.6843 | [3954, 1046] |
| 5 | Vitamin D | Single Linkage | 0.6690 | [4999, 1] |

### Algorithm Performance

| Algorithm | Max Score | Mean Score | Std Dev |
|-----------|-----------|------------|---------|
| Single Linkage | 0.7866 | 0.6032 | ±0.0861 |
| Average Linkage | 0.7424 | 0.5269 | ±0.0916 |
| Complete Linkage | 0.6969 | 0.4658 | ±0.1463 |
| K-Means | 0.6843 | 0.5008 | ±0.1491 |

---

## Key Findings

### 1. Why Triglycerides Performs Best

- Natural outliers (extremely high values)
- Right-skewed distribution creates clear separation
- Clinical threshold-based separation possible

### 2. Trade-off Identified

A fundamental trade-off exists between statistical performance and cluster balance:

- **High Scores (~0.79)**: Require extreme imbalance (4998:11 ratio)
- **Balanced Clusters (~0.56)**: Require algorithm compromise

### 3. Target Achievement

The 0.87+ target requires either:
- Extreme outlier exploitation (4998:2 ratio) - not clinically meaningful
- Artificial category creation - not pure unsupervised clustering

The theoretical maximum for natural clustering on continuous health data is approximately 0.79.

---

## Recommendations

### For Anomaly Detection
- **Metric**: Triglycerides
- **Algorithm**: Single Linkage
- **Use Case**: Identifying extreme values, rare conditions

### For Population Segmentation
- **Metrics**: BMI, Age, Blood Glucose
- **Algorithm**: K-Means
- **Use Case**: Identifying distinct health phenotypes

### For Risk Stratification
- **Metrics**: Blood Glucose, HbA1c, BMI
- **Algorithm**: K-Means with 3 clusters
- **Use Case**: Low/Medium/High risk classification

---

## Documentation

See `docs/PROJECT_DOCUMENTATION.md` for comprehensive methodology and results.

---

## Learning Outcomes

This project demonstrates:
- Systematic hyperparameter tuning methodology
- Comprehensive algorithm comparison
- Trade-off analysis between performance metrics
- Clinical interpretation of machine learning results
- Academic documentation standards

---

## References

1. Jain, A. K. (2010). Data clustering: 50 years beyond K-means. Pattern Recognition Letters.
2. Ward Jr, J. H. (1963). Hierarchical grouping to optimize an objective function. Journal of the American Statistical Association.
3. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics.
4. Calinski, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. Communications in Statistics.

---

## Author

**Cavin Otieno**
Master's of Science in Public Health Data Science
Advanced Machine Learning - Unit Project

---

*Project completed: January 2025*
