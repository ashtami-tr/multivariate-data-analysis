
# mat4203 - multivariate data analysis
# wine recognition dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from scipy.stats import chi2, f, pearsonr

# loading the wine dataset and put it into a dataframe

wine = load_wine()
df = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)
df["cultivar"] = wine.target

# overview of the dataset

print("WINE RECOGNITION DATASET")
print("\nfirst five rows:")
print(df.head())
print("\ndataset dimensions:")
print(df.shape)
print("\ndataset information:")
df.info()
print("\nsummary statistics:")
print(df.drop(columns="cultivar").describe().round(3))
print("\nsamples in each cultivar:")
print(df["cultivar"].value_counts().sort_index())

### TASK 1: exploratory multivariate analysis and normality check
# keep the chemical measurements separate from the cultivar labels

X = df.drop(columns="cultivar")
y = df["cultivar"]

# calculate the mean, covariance and correlation matrices

mean_vector = X.mean()
covariance_matrix = X.cov()
correlation_matrix = X.corr()

print("\nTASK 1: EXPLORATORY MULTIVARIATE ANALYSIS")
print("\nmean vector:")
print(mean_vector.round(3))
print("\ncovariance matrix:")
print(covariance_matrix.round(3))
print("\ncorrelation matrix:")
print(correlation_matrix.round(3))

# drawing a heatmap to make the correlations easier to compare

plt.figure(figsize=(12, 9))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0
)
plt.title("Figure 1: Correlation Matrix of Wine Chemical Measurements")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# checking multivariate normality using a chi-square q-q plot

X_array = X.to_numpy()
n, p = X_array.shape
mean_values = X_array.mean(axis=0)
cov_matrix = np.cov(X_array, rowvar=False)

# measuring how far each observation is, from the overall mean

centered_data = X_array - mean_values
cov_inverse = np.linalg.inv(cov_matrix)
mahalanobis_squared = np.einsum(
    "ij,jk,ik->i",
    centered_data,
    cov_inverse,
    centered_data
)

# comparing the distances with theoretical chi-square values

ordered_distances = np.sort(mahalanobis_squared)
probabilities = (np.arange(1, n + 1) - 0.5) / n
theoretical_values = chi2.ppf(probabilities, df=p)
plt.figure(figsize=(8, 6))
plt.scatter(
    theoretical_values,
    ordered_distances,
    alpha=0.7
)
lower = min(theoretical_values.min(), ordered_distances.min())
upper = max(theoretical_values.max(), ordered_distances.max())
plt.plot(
    [lower, upper],
    [lower, upper],
    "r--",
    label="reference line"
)

plt.xlabel("Theoretical Chi-square Quantiles (13 df)")
plt.ylabel("Ordered Squared Mahalanobis Distances")
plt.title("Figure 2: Chi-square Q-Q Plot for Multivariate Normality")
plt.legend()
plt.tight_layout()
plt.savefig("chi_square_qq_plot.png", dpi=300, bbox_inches="tight")
plt.show()


### TASK 2: hotelling's t-squared test and mahalanobis distance
# compare cultivar 0 and cultivar 1

group0 = df[df["cultivar"] == 0].drop(columns="cultivar")
group1 = df[df["cultivar"] == 1].drop(columns="cultivar")
X0 = group0.to_numpy()
X1 = group1.to_numpy()
n0, p = X0.shape
n1 = X1.shape[0]
mean0 = X0.mean(axis=0)
mean1 = X1.mean(axis=0)
cov0 = np.cov(X0, rowvar=False)
cov1 = np.cov(X1, rowvar=False)

# combining the two covariance matrices into a pooled covariance matrix

pooled_covariance = (
    (n0 - 1) * cov0 + (n1 - 1) * cov1
) / (n0 + n1 - 2)
mean_difference = mean0 - mean1
pooled_inverse = np.linalg.inv(pooled_covariance)

# calculating hotelling's t-squared statistic

T2 = (n0 * n1 / (n0 + n1)) * (
    mean_difference.T @ pooled_inverse @ mean_difference
)

# converting the result into an f-statistic to find the p-value

F_statistic = (
    (n0 + n1 - p - 1) /
    (p * (n0 + n1 - 2))
) * T2

df1 = p
df2 = n0 + n1 - p - 1
p_value = f.sf(F_statistic, df1, df2)
alpha = 0.05
print("\nTASK 2: HOTELLING'S T-SQUARED TEST")
print("cultivar 0 sample size:", n0)
print("cultivar 1 sample size:", n1)
print("\ncultivar 0 mean vector:")
print(pd.Series(mean0, index=group0.columns).round(3))
print("\ncultivar 1 mean vector:")
print(pd.Series(mean1, index=group1.columns).round(3))
print(f"\nt-squared statistic: {T2:.4f}")
print(f"f-statistic: {F_statistic:.4f}")
print(f"degrees of freedom: ({df1}, {df2})")
print(f"p-value: {p_value:.6g}")

if p_value < alpha:
    print("decision: reject the null hypothesis.")
    print("conclusion: the two cultivar mean vectors differ significantly.")
else:
    print("decision: fail to reject the null hypothesis.")
    print("conclusion: there is insufficient evidence of a difference.")

# calculating the mahalanobis distance between the two mean vectors

D_squared = mean_difference.T @ pooled_inverse @ mean_difference
D = np.sqrt(D_squared)
print("\nmahalanobis distance between cultivar means")
print(f"mahalanobis d-squared: {D_squared:.4f}")
print(f"mahalanobis distance: {D:.4f}")

# checking that the distance gives the same t-squared statistic

verified_T2 = (n0 * n1 / (n0 + n1)) * D_squared
print("\nverification:")
print(f"t-squared from distance: {verified_T2:.4f}")
print(f"hotelling's t-squared: {T2:.4f}")

### TASK 3: canonical correlation analysis
# choose two meaningful groups of chemical measurements

set_A_cols = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "proline"
]

set_B_cols = [
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280/od315_of_diluted_wines"
]

A = df[set_A_cols].to_numpy()
B = df[set_B_cols].to_numpy()

# standardize the variables before running cca

scaler_A = StandardScaler()
scaler_B = StandardScaler()
A_scaled = scaler_A.fit_transform(A)
B_scaled = scaler_B.fit_transform(B)
print("\nTASK 3: CANONICAL CORRELATION ANALYSIS")
print("number of observations:", len(df))
print("number of variables in set A:", len(set_A_cols))
print("number of variables in set B:", len(set_B_cols))

# find the canonical variates and their correlations

n_components = min(A_scaled.shape[1], B_scaled.shape[1])
cca = CCA(
    n_components=n_components,
    max_iter=2000
)
U, V = cca.fit_transform(A_scaled, B_scaled)
canonical_correlations = []
for i in range(n_components):
    r, _ = pearsonr(U[:, i], V[:, i])
    canonical_correlations.append(r)
print("\ncanonical correlations:")
for i, r in enumerate(canonical_correlations, start=1):
    print(f"canonical pair {i}: {r:.4f}")

# test the significance of each remaining set of canonical correlations

N = A_scaled.shape[0]
p_A = A_scaled.shape[1]
p_B = B_scaled.shape[1]
print("\ncanonical correlation significance tests")
for k in range(n_components):
    remaining_correlations = np.array(canonical_correlations[k:])
    wilks_lambda = np.prod(1 - remaining_correlations**2)

    # use bartlett's approximation for the chi-square test

    test_df = (p_A - k) * (p_B - k)
    factor = N - 1 - (p_A + p_B + 1) / 2
    chi_square_stat = -factor * np.log(wilks_lambda)
    test_p_value = chi2.sf(chi_square_stat, test_df)
    print(f"\ntest starting at canonical pair {k + 1}")
    print(f"wilks' lambda: {wilks_lambda:.6f}")
    print(f"chi-square statistic: {chi_square_stat:.4f}")
    print(f"degrees of freedom: {test_df}")
    print(f"p-value: {test_p_value:.6g}")
    if test_p_value < 0.05:
        print("decision: reject the null hypothesis.")
    else:
        print("decision: fail to reject the null hypothesis.")

# look at the weights and loadings for the first canonical pair

weights_A = pd.Series(
    cca.x_weights_[:, 0],
    index=set_A_cols,
    name="set A coefficient"
)
weights_B = pd.Series(
    cca.y_weights_[:, 0],
    index=set_B_cols,
    name="set B coefficient"
)

# loadings show how closely each original variable relates to its variate

loadings_A = pd.Series(
    [
        np.corrcoef(A_scaled[:, j], U[:, 0])[0, 1]
        for j in range(A_scaled.shape[1])
    ],
    index=set_A_cols,
    name="set A loading"
)
loadings_B = pd.Series(
    [
        np.corrcoef(B_scaled[:, j], V[:, 0])[0, 1]
        for j in range(B_scaled.shape[1])
    ],
    index=set_B_cols,
    name="set B loading"
)

print("\nfirst canonical pair correlation:")
print(f"{canonical_correlations[0]:.4f}")
print("\nset A: first canonical coefficients")
print(weights_A.round(4))
print("\nset B: first canonical coefficients")
print(weights_B.round(4))
print("\nset A: first canonical loadings")
print(loadings_A.round(4))
print("\nset B: first canonical loadings")
print(loadings_B.round(4))

### TASK 4: principal component analysis
# use the chemical measurements and standardize

X = df.drop(columns="cultivar")
y = df["cultivar"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# fit pca and keep all components to check their variance

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance = pca.explained_variance_ratio_
first_two_variance = explained_variance[:2].sum()
print("\nTASK 4: PRINCIPAL COMPONENT ANALYSIS")
print("original data shape:", X.shape)
print("pca-transformed data shape:", X_pca.shape)
print("\nexplained variance ratio:")
for i, ratio in enumerate(explained_variance, start=1):
    print(f"pc{i}: {ratio:.4f} ({ratio * 100:.2f}%)")
print(
    "\nvariance explained by the first two components:",
    f"{first_two_variance * 100:.2f}%"
)

# ploting the wine samples using the first two principal components

pc1_percent = explained_variance[0] * 100
pc2_percent = explained_variance[1] * 100
plt.figure(figsize=(8, 6))
for cultivar in sorted(y.unique()):
    mask = y == cultivar

    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cultivar {cultivar}",
        alpha=0.75
    )

plt.xlabel(f"Principal Component 1 ({pc1_percent:.2f}%)")
plt.ylabel(f"Principal Component 2 ({pc2_percent:.2f}%)")
plt.title("Figure 3: PCA Scatter Plot of Wine Cultivars")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_scatter_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# biplot to show how the original features relate to the components

loadings = pca.components_.T[:, :2]
plt.figure(figsize=(10, 8))
for cultivar in sorted(y.unique()):
    mask = y == cultivar
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        label=f"Cultivar {cultivar}",
        alpha=0.55
    )

# adding an arrow for each chemical measurement

arrow_scale = 3.0
for i, feature in enumerate(X.columns):
    x_loading = loadings[i, 0] * arrow_scale
    y_loading = loadings[i, 1] * arrow_scale
    plt.arrow(
        0,
        0,
        x_loading,
        y_loading,
        color="red",
        alpha=0.75,
        head_width=0.08,
        length_includes_head=True
    )

    plt.text(
        x_loading * 1.10,
        y_loading * 1.10,
        feature,
        color="darkred",
        fontsize=8,
        ha="center"
    )

plt.xlabel(f"Principal Component 1 ({pc1_percent:.2f}%)")
plt.ylabel(f"Principal Component 2 ({pc2_percent:.2f}%)")
plt.title("Figure 4: PCA Biplot of Wine Dataset")
plt.axhline(0, color="gray", linewidth=0.6)
plt.axvline(0, color="gray", linewidth=0.6)
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("pca_biplot.png", dpi=300, bbox_inches="tight")
plt.show()

# printing the loadings to compare the contribution of each feature

loading_table = pd.DataFrame(
    loadings,
    index=X.columns,
    columns=["PC1 Loading", "PC2 Loading"]
)
print("\npca loadings for the first two components:")
print(loading_table.round(4))

# analysis completion

print("\nanalysis finished.")
print("the plots have been saved in the project folder.")