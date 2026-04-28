#!/usr/bin/env python3
"""
Exploratory Data Analysis Script
Performs comprehensive EDA on the Brazilian E-commerce dataset
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Load the master dataset
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'processed' / 'master_dataset.csv'

df = pd.read_csv(data_path)

print(f"\nDataset Shape: {df.shape}")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

## 1. Dataset Overview
print("\n" + "=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)

print(f"\nShape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"\nData Types:")
print(df.dtypes.value_counts())

print("\nMISSING VALUES:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)

print(missing_df[missing_df['Missing Count'] > 0].head(10))

print("\nSAMPLE DATA (first 5 rows):")
print(df.head())

## 2. Numerical Analysis
print("\n" + "=" * 70)
print("2. NUMERICAL VARIABLES ANALYSIS")
print("=" * 70)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"Categorical columns: {len(categorical_cols)}")

print("\nNUMERICAL SUMMARY STATISTICS:")
print(df[numerical_cols].describe())

# Key numerical variables for visualization
key_numerical = ['price', 'freight_value', 'item_total_value', 'delivery_time_days', 
                'avg_review_score', 'total_payment_value', 'product_weight_g']
key_numerical = [col for col in key_numerical if col in df.columns]

print(f"\nKey numerical variables: {key_numerical}")

# Create distribution plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for i, col in enumerate(key_numerical[:9]):
    if col in df.columns:
        df[col].hist(bins=50, ax=axes[i], alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

# Remove empty subplots
for j in range(len(key_numerical), 9):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(project_root / 'reports' / 'eda_distributions.png', dpi=100, bbox_inches='tight')
print(f"\nSaved distribution plots to reports/eda_distributions.png")
plt.close()

## 3. Categorical Analysis
print("\n" + "=" * 70)
print("3. CATEGORICAL VARIABLES ANALYSIS")
print("=" * 70)

key_categorical = ['order_status', 'customer_state', 'product_category_name_english', 
                  'dominant_payment_type', 'dominant_sentiment', 'order_stage']
key_categorical = [col for col in key_categorical if col in df.columns]

for col in key_categorical[:3]:  # Limit to first 3 for brevity
    if col in df.columns:
        print(f"\n{col.upper()} VALUE COUNTS:")
        value_counts = df[col].value_counts()
        print(value_counts.head(10))

## 4. Correlation Analysis
print("\n" + "=" * 70)
print("4. CORRELATION ANALYSIS")
print("=" * 70)

# Select key numerical columns for correlation
corr_cols = ['price', 'freight_value', 'delivery_time_days', 'avg_review_score', 
            'total_payment_value', 'product_weight_g', 'order_item_count']
corr_cols = [col for col in corr_cols if col in df.columns]

if len(corr_cols) > 1:
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', linewidths=1)
    plt.title('Correlation Matrix of Key Numerical Variables')
    plt.tight_layout()
    plt.savefig(project_root / 'reports' / 'eda_correlation.png', dpi=100, bbox_inches='tight')
    print(f"\nSaved correlation matrix to reports/eda_correlation.png")
    plt.close()
    
    # Print strong correlations
    print("\nSTRONG CORRELATIONS (|r| > 0.5):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                print(f"  {var1} ↔ {var2}: {corr_val:.3f}")

## 5. Geographic Analysis
print("\n" + "=" * 70)
print("5. GEOGRAPHIC ANALYSIS")
print("=" * 70)

if 'customer_state' in df.columns:
    state_analysis = df.groupby('customer_state').agg({
        'order_id': 'count',
        'total_payment_value': ['mean', 'sum'],
        'avg_review_score': 'mean',
        'delivery_time_days': 'mean'
    }).round(2)
    
    state_analysis.columns = ['Order_Count', 'Avg_Payment', 'Total_Revenue', 
                             'Avg_Review_Score', 'Avg_Delivery_Days']
    state_analysis = state_analysis.sort_values('Order_Count', ascending=False)
    
    print("\nTOP 10 STATES BY ORDER COUNT:")
    print(state_analysis.head(10))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    state_analysis.head(10)['Order_Count'].plot(kind='bar', ax=axes[0,0], color='steelblue')
    axes[0,0].set_title('Top 10 States by Order Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    state_analysis.head(10)['Total_Revenue'].plot(kind='bar', ax=axes[0,1], color='green')
    axes[0,1].set_title('Top 10 States by Total Revenue')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    state_analysis.head(10)['Avg_Review_Score'].plot(kind='bar', ax=axes[1,0], color='orange')
    axes[1,0].set_title('Top 10 States by Average Review Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    state_analysis.head(10)['Avg_Delivery_Days'].plot(kind='bar', ax=axes[1,1], color='red')
    axes[1,1].set_title('Top 10 States by Average Delivery Time')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(project_root / 'reports' / 'eda_geographic.png', dpi=100, bbox_inches='tight')
    print(f"\nSaved geographic analysis to reports/eda_geographic.png")
    plt.close()

## 6. Product Analysis
print("\n" + "=" * 70)
print("6. PRODUCT CATEGORY ANALYSIS")
print("=" * 70)

if 'product_category_name_english' in df.columns:
    category_analysis = df.groupby('product_category_name_english').agg({
        'order_id': 'count',
        'price': 'mean',
        'total_payment_value': 'sum',
        'avg_review_score': 'mean',
        'freight_value': 'mean'
    }).round(2)
    
    category_analysis.columns = ['Order_Count', 'Avg_Price', 'Total_Revenue', 
                                'Avg_Review_Score', 'Avg_Freight']
    category_analysis = category_analysis.sort_values('Order_Count', ascending=False)
    
    print("\nTOP 15 PRODUCT CATEGORIES:")
    print(category_analysis.head(15))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    category_analysis.head(15)['Order_Count'].plot(kind='barh', ax=axes[0,0], color='steelblue')
    axes[0,0].set_title('Top 15 Categories by Order Count')
    
    category_analysis.head(15)['Total_Revenue'].plot(kind='barh', ax=axes[0,1], color='green')
    axes[0,1].set_title('Top 15 Categories by Total Revenue')
    
    category_analysis.head(15)['Avg_Price'].plot(kind='barh', ax=axes[1,0], color='purple')
    axes[1,0].set_title('Top 15 Categories by Average Price')
    
    category_analysis.head(15)['Avg_Review_Score'].plot(kind='barh', ax=axes[1,1], color='orange')
    axes[1,1].set_title('Top 15 Categories by Average Review Score')
    
    plt.tight_layout()
    plt.savefig(project_root / 'reports' / 'eda_products.png', dpi=100, bbox_inches='tight')
    print(f"\nSaved product analysis to reports/eda_products.png")
    plt.close()

## 7. Time Series Analysis
print("\n" + "=" * 70)
print("7. TIME SERIES ANALYSIS")
print("=" * 70)

if 'order_purchase_timestamp' in df.columns:
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
    df['order_year_month'] = df['order_purchase_timestamp'].dt.to_period('M')
    
    monthly_trends = df.groupby('order_year_month').agg({
        'order_id': 'count',
        'total_payment_value': ['sum', 'mean'],
        'avg_review_score': 'mean'
    })
    
    monthly_trends.columns = ['Order_Count', 'Total_Revenue', 'Avg_Order_Value', 'Avg_Review_Score']
    
    print("\nMONTHLY TRENDS:")
    print(monthly_trends.tail(12))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    monthly_trends['Order_Count'].plot(ax=axes[0,0], color='steelblue', linewidth=2)
    axes[0,0].set_title('Monthly Order Count Trend')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    monthly_trends['Total_Revenue'].plot(ax=axes[0,1], color='green', linewidth=2)
    axes[0,1].set_title('Monthly Revenue Trend')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    monthly_trends['Avg_Order_Value'].plot(ax=axes[1,0], color='purple', linewidth=2)
    axes[1,0].set_title('Monthly Average Order Value Trend')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    monthly_trends['Avg_Review_Score'].plot(ax=axes[1,1], color='orange', linewidth=2)
    axes[1,1].set_title('Monthly Average Review Score Trend')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(project_root / 'reports' / 'eda_timeseries.png', dpi=100, bbox_inches='tight')
    print(f"\nSaved time series analysis to reports/eda_timeseries.png")
    plt.close()

## 8. Key Business Insights
print("\n" + "=" * 70)
print("KEY BUSINESS INSIGHTS")
print("=" * 70)

total_orders = df['order_id'].nunique()
total_customers = df['customer_unique_id'].nunique() if 'customer_unique_id' in df.columns else 'N/A'
total_revenue = df['total_payment_value'].sum() if 'total_payment_value' in df.columns else 'N/A'
avg_order_value = df['total_payment_value'].mean() if 'total_payment_value' in df.columns else 'N/A'
avg_review_score = df['avg_review_score'].mean() if 'avg_review_score' in df.columns else 'N/A'

print(f"\nBUSINESS OVERVIEW:")
print(f"   • Total Orders: {total_orders:,}")
if total_customers != 'N/A':
    print(f"   • Total Customers: {total_customers:,}")
if total_revenue != 'N/A':
    print(f"   • Total Revenue: R$ {total_revenue:,.2f}")
if avg_order_value != 'N/A':
    print(f"   • Average Order Value: R$ {avg_order_value:.2f}")
if avg_review_score != 'N/A':
    print(f"   • Average Review Score: {avg_review_score:.2f}/5")

if 'product_category_name_english' in df.columns:
    top_category = df['product_category_name_english'].value_counts().index[0]
    print(f"\n   • Top Performing Category: {top_category}")

if 'customer_state' in df.columns:
    top_state = df['customer_state'].value_counts().index[0]
    print(f"   • Top State by Orders: {top_state}")

if 'dominant_payment_type' in df.columns:
    top_payment = df['dominant_payment_type'].value_counts().index[0]
    print(f"   • Most Popular Payment Method: {top_payment}")

if 'delivery_time_days' in df.columns:
    avg_delivery = df['delivery_time_days'].mean()
    print(f"   • Average Delivery Time: {avg_delivery:.1f} days")

print("\n" + "=" * 70)
print("✓ EDA COMPLETED SUCCESSFULLY!")
print("=" * 70)
