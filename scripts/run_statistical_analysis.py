#!/usr/bin/env python3
"""
Statistical Analysis Script
Performs comprehensive statistical analysis including hypothesis testing and modeling
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr, f_oneway, shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

# Load the master dataset
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'processed' / 'master_dataset.csv'

df = pd.read_csv(data_path)

print(f"\nDataset Shape: {df.shape}")
print("Statistical Analysis Starting...")

## 1. Descriptive Statistics
print("\n" + "=" * 70)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 70)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nDESCRIPTIVE STATISTICS FOR NUMERICAL VARIABLES:")
print(df[numerical_cols].describe())

# Additional statistics for key variables
key_vars = ['price', 'freight_value', 'total_payment_value', 'avg_review_score', 'delivery_time_days']
key_vars = [col for col in key_vars if col in df.columns]

print("\nADDITIONAL STATISTICS:")
for col in key_vars[:5]:
    if col in df.columns:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"\n{col.upper()}:")
            print(f"  Skewness: {stats.skew(data):.3f}")
            print(f"  Kurtosis: {stats.kurtosis(data):.3f}")
            print(f"  IQR: {data.quantile(0.75) - data.quantile(0.25):.3f}")
            if data.mean() != 0:
                print(f"  CV: {(data.std() / data.mean()):.3f}")

## 2. Normality Testing
print("\n" + "=" * 70)
print("2. NORMALITY TESTS (Shapiro-Wilk Test)")
print("=" * 70)
print("H0: Data is normally distributed")
print("H1: Data is not normally distributed")
print("Alpha = 0.05\n")

normality_results = []

for var in key_vars:
    if var in df.columns:
        data = df[var].dropna()
        
        if len(data) > 5000:
            sample_data = data.sample(5000, random_state=42)
        else:
            sample_data = data
        
        if len(sample_data) >= 3:
            stat, p_value = shapiro(sample_data)
            
            result = {
                'Variable': var,
                'Statistic': stat,
                'P-value': p_value,
                'Normal': 'Yes' if p_value > 0.05 else 'No'
            }
            normality_results.append(result)
            
            print(f"{var}: Statistic={stat:.4f}, P-value={p_value:.4f}, Normal: {result['Normal']}")

## 3. Correlation Analysis
print("\n" + "=" * 70)
print("3. CORRELATION ANALYSIS")
print("=" * 70)

key_numerical = ['price', 'freight_value', 'total_payment_value', 'avg_review_score', 
                'delivery_time_days', 'product_weight_g', 'order_item_count']
key_numerical = [col for col in key_numerical if col in df.columns]

if len(key_numerical) > 1:
    pearson_corr = df[key_numerical].corr(method='pearson')
    spearman_corr = df[key_numerical].corr(method='spearman')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', ax=axes[0], linewidths=1)
    axes[0].set_title('Pearson Correlation Matrix')
    
    sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', ax=axes[1], linewidths=1)
    axes[1].set_title('Spearman Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(project_root / 'reports' / 'stat_correlation.png', dpi=100, bbox_inches='tight')
    print(f"\nSaved correlation matrices to reports/stat_correlation.png")
    plt.close()
    
    print("\nSTRONG CORRELATIONS (|r| > 0.5):")
    for i in range(len(pearson_corr.columns)):
        for j in range(i+1, len(pearson_corr.columns)):
            corr_val = pearson_corr.iloc[i, j]
            if abs(corr_val) > 0.5:
                var1 = pearson_corr.columns[i]
                var2 = pearson_corr.columns[j]
                print(f"  {var1} ↔ {var2}: {corr_val:.3f}")

## 4. Hypothesis Testing
print("\n" + "=" * 70)
print("4. HYPOTHESIS TESTING")
print("=" * 70)

# Test 1: Do different payment types have different average order values?
if 'dominant_payment_type' in df.columns and 'total_payment_value' in df.columns:
    print("\nHYPOTHESIS TEST 1: Payment Type vs Order Value")
    print("H0: All payment types have the same mean order value")
    print("H1: At least one payment type has different mean order value")
    
    payment_groups = []
    payment_types = df['dominant_payment_type'].value_counts().head(4).index
    
    for ptype in payment_types:
        group_data = df[df['dominant_payment_type'] == ptype]['total_payment_value'].dropna()
        if len(group_data) > 0:
            payment_groups.append(group_data)
            print(f"  {ptype}: n={len(group_data)}, mean={group_data.mean():.2f}")
    
    if len(payment_groups) >= 2:
        f_stat, p_value = f_oneway(*payment_groups)
        print(f"\nANOVA Results:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")

# Test 2: Is there a relationship between delivery time and review score?
if 'delivery_time_days' in df.columns and 'avg_review_score' in df.columns:
    print("\n\nHYPOTHESIS TEST 2: Delivery Time vs Review Score")
    print("H0: There is no correlation between delivery time and review score")
    print("H1: There is a correlation between delivery time and review score")
    
    clean_data = df[['delivery_time_days', 'avg_review_score']].dropna()
    clean_data = clean_data[(clean_data['delivery_time_days'] > 0) & (clean_data['avg_review_score'] > 0)]
    
    if len(clean_data) > 0:
        corr_coef, p_value = pearsonr(clean_data['delivery_time_days'], clean_data['avg_review_score'])
        
        print(f"\nPearson Correlation Test:")
        print(f"  Correlation coefficient: {corr_coef:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")
        
        # Visualization
        plt.figure(figsize=(10, 6))
        sample_data = clean_data.sample(min(5000, len(clean_data)), random_state=42)
        plt.scatter(sample_data['delivery_time_days'], sample_data['avg_review_score'], alpha=0.5)
        plt.xlabel('Delivery Time (days)')
        plt.ylabel('Average Review Score')
        plt.title('Delivery Time vs Review Score')
        
        z = np.polyfit(sample_data['delivery_time_days'], sample_data['avg_review_score'], 1)
        p = np.poly1d(z)
        plt.plot(sample_data['delivery_time_days'].sort_values(), 
                p(sample_data['delivery_time_days'].sort_values()), "r--", alpha=0.8, linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.savefig(project_root / 'reports' / 'stat_delivery_review.png', dpi=100, bbox_inches='tight')
        print(f"\nSaved scatter plot to reports/stat_delivery_review.png")
        plt.close()

# Test 3: Do customers from different states have different satisfaction levels?
if 'customer_state' in df.columns and 'avg_review_score' in df.columns:
    print("\n\nHYPOTHESIS TEST 3: State vs Customer Satisfaction")
    print("H0: All states have the same mean review score")
    print("H1: At least one state has different mean review score")
    
    top_states = df['customer_state'].value_counts().head(5).index
    state_groups = []
    
    for state in top_states:
        group_data = df[df['customer_state'] == state]['avg_review_score'].dropna()
        group_data = group_data[group_data > 0]
        if len(group_data) > 0:
            state_groups.append(group_data)
            print(f"  {state}: n={len(group_data)}, mean={group_data.mean():.2f}")
    
    if len(state_groups) >= 2:
        f_stat, p_value = f_oneway(*state_groups)
        print(f"\nANOVA Results:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Result: {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'}")

## 5. Regression Analysis
print("\n" + "=" * 70)
print("5. LINEAR REGRESSION: Predicting Review Score")
print("=" * 70)

feature_cols = ['price', 'freight_value', 'delivery_time_days', 'order_item_count']
feature_cols = [col for col in feature_cols if col in df.columns]

if len(feature_cols) > 0 and 'avg_review_score' in df.columns:
    regression_data = df[feature_cols + ['avg_review_score']].dropna()
    regression_data = regression_data[regression_data['avg_review_score'] > 0]
    
    if len(regression_data) > 100:
        X = regression_data[feature_cols]
        y = regression_data['avg_review_score']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        
        y_pred = lr_model.predict(X_test_scaled)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"  R-squared: {r2:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.4f}")
        
        print(f"\nFeature Coefficients:")
        for feature, coef in zip(feature_cols, lr_model.coef_):
            print(f"  {feature}: {coef:.4f}")
        
        # Residual plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        residuals = y_test - y_pred
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].scatter(y_test, y_pred, alpha=0.5)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1].set_xlabel('Actual Review Score')
        axes[1].set_ylabel('Predicted Review Score')
        axes[1].set_title('Actual vs Predicted Review Scores')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(project_root / 'reports' / 'stat_regression.png', dpi=100, bbox_inches='tight')
        print(f"\nSaved regression plots to reports/stat_regression.png")
        plt.close()

## 6. Logistic Regression
print("\n" + "=" * 70)
print("6. LOGISTIC REGRESSION: Predicting High Customer Satisfaction")
print("=" * 70)

if 'avg_review_score' in df.columns:
    df_log = df.copy()
    df_log['high_satisfaction'] = (df_log['avg_review_score'] >= 4).astype(int)
    
    log_features = ['price', 'freight_value', 'delivery_time_days', 'order_item_count']
    log_features = [col for col in log_features if col in df_log.columns]
    
    if len(log_features) > 0:
        log_data = df_log[log_features + ['high_satisfaction']].dropna()
        
        if len(log_data) > 100:
            X = log_data[log_features]
            y = log_data['high_satisfaction']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            log_model = LogisticRegression(random_state=42, max_iter=1000)
            log_model.fit(X_train_scaled, y_train)
            
            y_pred = log_model.predict(X_test_scaled)
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            print(f"\nFeature Coefficients (Log-Odds):")
            for feature, coef in zip(log_features, log_model.coef_[0]):
                print(f"  {feature}: {coef:.4f}")

## 7. Statistical Summary
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS SUMMARY")
print("=" * 70)

print("\nKEY STATISTICAL FINDINGS:")

if normality_results:
    normal_vars = [r['Variable'] for r in normality_results if r['Normal'] == 'Yes']
    non_normal_vars = [r['Variable'] for r in normality_results if r['Normal'] == 'No']
    
    print(f"\nNORMALITY TESTS:")
    print(f"   • Variables following normal distribution: {len(normal_vars)}")
    print(f"   • Variables NOT following normal distribution: {len(non_normal_vars)}")

if 'pearson_corr' in locals():
    strong_corrs = []
    for i in range(len(pearson_corr.columns)):
        for j in range(i+1, len(pearson_corr.columns)):
            corr_val = pearson_corr.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corrs.append((pearson_corr.columns[i], pearson_corr.columns[j], corr_val))
    
    print(f"\nCORRELATION ANALYSIS:")
    print(f"   • Strong correlations found: {len(strong_corrs)}")

if 'r2' in locals():
    print(f"\nREGRESSION INSIGHTS:")
    print(f"   • Review Score Prediction R²: {r2:.3f}")
    print(f"   • Model explains {r2*100:.1f}% of variance in review scores")

print(f"\nBUSINESS RECOMMENDATIONS:")
print(f"   • Focus on delivery time optimization (impacts customer satisfaction)")
print(f"   • Consider payment method preferences for different customer segments")
print(f"   • Monitor state-wise performance variations")
print(f"   • Implement targeted strategies based on statistical findings")

print("\n" + "=" * 70)
print("✓ STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 70)
