# Multivariate Data Analysis of Wine

## About the Project

This project was done as part of my Multivariate Data Analysis course. I used the Wine Recognition dataset to explore the chemical properties of wine and understand how different statistical methods can be used to analyse multivariate data.

The main goal was to study the relationships between the chemical measurements, compare wine cultivars, and reduce the number of variables while keeping as much useful information as possible.

## Dataset

I used the Wine Recognition dataset available through scikit-learn. It contains 178 wine samples, with 13 chemical measurements for each sample.

The dataset has three wine cultivars, which are labelled as 0, 1, and 2.

## What I Did

### Task 1: Exploratory Multivariate Analysis

I calculated the mean vector, covariance matrix, and correlation matrix to understand the data. I also created a correlation heatmap and used a chi-square Q-Q plot to check whether the data followed a multivariate normal distribution.

### Task 2: Comparing Wine Cultivars

I compared Cultivar 0 and Cultivar 1 using Hotelling's T-squared test. I also calculated the Mahalanobis distance between their mean vectors to measure how different the two groups were.

### Task 3: Canonical Correlation Analysis

I divided the chemical measurements into two groups and used canonical correlation analysis to study the relationship between them. I calculated the canonical correlations, tested their significance, and examined the variables that contributed most to the first canonical pair.

### Task 4: Principal Component Analysis

I used PCA to reduce the 13 chemical measurements to a smaller number of components. I checked how much variance each component explained and created a scatter plot and biplot to understand the separation between the three wine cultivars.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

## Files in This Project

- `main.py` contains the Python code for the analysis.
- `correlation_heatmap.png` shows the correlations between the chemical measurements.
- `chi_square_qq_plot.png` shows the multivariate normality check.
- `pca_scatter_plot.png` shows the wine samples using the first two principal components.
- `pca_biplot.png` shows the wine samples along with the chemical feature loadings.

## How to Run the Project

1. Make sure Python is installed on your computer.
2. Install the required libraries using the command below.

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

3. Download or clone this repository.
4. Open the project folder in VS Code.
5. Run the Python file.

```bash
python main.py
```

The program prints the analysis results and displays the plots. It also saves the figures in the project folder.

## What I Learned

Through this project, I got practical experience with multivariate statistical methods and learned how to apply them using Python. I also learned how to interpret statistical results and use visualisations to understand patterns in data.

## Reference

UCI Machine Learning Repository. Wine Recognition dataset.

https://archive.ics.uci.edu/dataset/109/wine
