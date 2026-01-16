# Alternative Clustering Methods - Final Results

## Summary

Testing various clustering algorithms to achieve better Silhouette Scores than K-Means baseline (0.5590).

## Results Comparison

| Method | Silhouette Score | Clusters | Balance | Improvement |
|--------|------------------|----------|---------|-------------|
| **Single Linkage Hierarchical** | **0.6533** | 2 | Unbalanced | +16.87% ✓ |
| GMM (Full Covariance) | 0.5588 | 2 | Balanced | -0.04% |
| K-Means (Baseline) | 0.5590 | 2 | Balanced | baseline |
| Ward Linkage | 0.5580 | 2 | Balanced | -0.18% |
| Complete Linkage | 0.5289 | 2 | Balanced | -5.39% |
| Average Linkage | 0.5084 | 3 | Balanced | -9.05% |

## Best Methods

### 1. Single Linkage Hierarchical Clustering (Best Score: 0.6533)
- **Score**: 0.6533 (+16.87% improvement over K-Means)
- **Clusters**: 2
- **Distribution**: 4,998 vs 2 samples (99.96% vs 0.04%)
- **Note**: Achieves highest score but creates one dominant cluster

**Use Case**: Best when you only care about Silhouette Score and cluster imbalance is acceptable

**Code**:
```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=2, linkage='single')
labels = model.fit_predict(X_scaled)
score = silhouette_score(X_scaled, labels)  # 0.6533
```

### 2. GMM (Best Balanced Score: 0.5588)
- **Score**: 0.5588 (similar to K-Means)
- **Clusters**: 2
- **Distribution**: Balanced
- **Note**: Best for when you need probabilistic clustering

**Code**:
```python
from sklearn.mixture import GaussianMixture

model = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
labels = model.fit_predict(X_scaled)
score = silhouette_score(X_scaled, labels)  # 0.5588
```

### 3. Ward Linkage (Good Balanced: 0.5580)
- **Score**: 0.5580
- **Clusters**: 2
- **Distribution**: Balanced (46.3% / 53.7%)
- **Note**: Minimizes within-cluster variance

**Code**:
```python
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = model.fit_predict(X_scaled)
score = silhouette_score(X_scaled, labels)  # 0.5580
```

## Key Findings

1. **Single Linkage achieves highest score** (0.6533) but creates extreme cluster imbalance
2. **For balanced clusters**, all methods cluster around 0.55-0.56, similar to K-Means
3. **K-Means remains optimal** for balanced, well-separated clusters
4. **Target 0.87-1.00** still not achievable with natural clustering

## Recommendations

| Goal | Recommended Method |
|------|-------------------|
| Maximize Silhouette Score | Single Linkage Hierarchical |
| Balanced Clusters + Good Score | K-Means or GMM |
| Minimized Variance | Ward Linkage |
| Probabilistic Assignment | GMM |

## Conclusion

**Best Overall**: Agglomerative Hierarchical Clustering with Single Linkage achieves 0.6533, which is **16.87% better than K-Means**.

**Trade-off**: Single Linkage creates one dominant cluster (99.96%) with only 2 outlier points in the second cluster.

For practical health phenotype discovery where balanced clusters are needed, **K-Means remains optimal** with a score of 0.5590.
