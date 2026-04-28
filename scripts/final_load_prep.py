"""
Final data preparation and optimization for Tableau.

This script performs final cleaning, type optimization, and preparation
of the master dataset for Tableau visualization and analysis.

Key improvements:
- Proper datetime conversion for all date columns
- Intelligent handling of missing values
- Data type optimization (reduces file size)
- Removal of duplicate rows
- Creation of additional derived features
- Standardization and validation
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("final_load_prep")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def load_data(input_path: Path) -> pd.DataFrame:
    """Load the master dataset."""
    LOGGER.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    LOGGER.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    original_len = len(df)
    df = df.drop_duplicates()
    removed = original_len - len(df)
    if removed > 0:
        LOGGER.info(f"Removed {removed:,} duplicate rows")
    return df


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp columns to datetime format."""
    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            LOGGER.info(f"Converted {col} to datetime")
    
    return df


def optimize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize numeric column types to reduce file size."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        # Check if column can be int
        if df[col].dtype == 'float64':
            # If mostly ints with few nulls, convert to Int64 (nullable int)
            if df[col].notna().sum() > 0:
                if (df[col].dropna() == df[col].dropna().astype(int)).all():
                    df[col] = df[col].astype('Int64')
                else:
                    # Keep as float32 for floating point numbers
                    df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            # Check if can fit in int32
            if df[col].min() >= -2147483648 and df[col].max() <= 2147483647:
                df[col] = df[col].astype('int32')
    
    LOGGER.info("Optimized numeric column types")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values intelligently."""
    
    # For delivery-related columns, fill with 'not_applicable' or 0 for orders not delivered
    delivery_cols = [
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'delivery_time_days',
        'delivery_delay_vs_estimate_days',
        'actual_delivery_days'
    ]
    
    for col in delivery_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if col in ['delivery_time_days', 'delivery_delay_vs_estimate_days', 'actual_delivery_days']:
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(pd.NaT, inplace=True)
            LOGGER.info(f"Handled {null_count:,} missing values in {col}")
    
    # For review columns, fill with 0 or 'no_review'
    review_cols = [
        'avg_review_score',
        'min_review_score',
        'max_review_score',
        'review_row_count',
        'negative_review_count'
    ]
    
    for col in review_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            df[col].fillna(0, inplace=True)
            LOGGER.info(f"Filled {null_count:,} missing values in {col} with 0")
    
    # For dominant_sentiment, fill with 'no_review'
    if 'dominant_sentiment' in df.columns:
        null_count = df['dominant_sentiment'].isnull().sum()
        df['dominant_sentiment'].fillna('no_review', inplace=True)
        LOGGER.info(f"Filled {null_count:,} missing values in dominant_sentiment with 'no_review'")
    
    # For payment columns, fill with 0
    payment_cols = [
        'total_payment_value',
        'payment_row_count',
        'payment_method_count',
        'average_payment_installments',
        'average_payment_value_per_installment',
        'credit_card_payment_rows',
        'full_payment_rows'
    ]
    
    for col in payment_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                df[col].fillna(0, inplace=True)
                LOGGER.info(f"Filled {null_count:,} missing values in {col} with 0")
    
    # For dominant_payment_type, fill with 'unknown'
    if 'dominant_payment_type' in df.columns:
        null_count = df['dominant_payment_type'].isnull().sum()
        if null_count > 0:
            df['dominant_payment_type'].fillna('unknown', inplace=True)
            LOGGER.info(f"Filled {null_count:,} missing values in dominant_payment_type")
    
    # Fill remaining nulls in text columns
    text_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in text_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            df[col].fillna('unknown', inplace=True)
    
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional derived features for better analysis."""
    
    # Order value metrics
    if 'order_items_total_value' in df.columns and 'order_items_total_freight' in df.columns:
        df['order_total_with_freight'] = df['order_items_total_value'] + df['order_items_total_freight']
        LOGGER.info("Created order_total_with_freight feature")
    
    # Shipping efficiency: freight as percentage of order value
    if 'order_items_total_value' in df.columns and 'order_items_total_freight' in df.columns:
        df['shipping_cost_percentage'] = (
            df['order_items_total_freight'] / df['order_items_total_value'].replace(0, 1) * 100
        ).round(2)
        LOGGER.info("Created shipping_cost_percentage feature")
    
    # Price variance in orders
    if 'max_item_price' in df.columns and 'min_item_price' in df.columns:
        df['item_price_variance'] = df['max_item_price'] - df['min_item_price']
        LOGGER.info("Created item_price_variance feature")
    
    # Payment concentration (single method dominance)
    if 'payment_method_count' in df.columns and 'payment_row_count' in df.columns:
        df['payment_concentration'] = (
            df['credit_card_payment_rows'] / df['payment_row_count'].replace(0, 1) * 100
        ).round(2)
        df['payment_concentration'].fillna(0, inplace=True)
        LOGGER.info("Created payment_concentration feature")
    
    # Order complexity score (number of items × number of sellers)
    if 'order_item_count' in df.columns and 'distinct_sellers' in df.columns:
        df['order_complexity'] = df['order_item_count'] * df['distinct_sellers']
        LOGGER.info("Created order_complexity feature")
    
    # Extract month and year from order purchase
    if 'order_purchase_timestamp' in df.columns:
        df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')
        df['order_year'] = df['order_purchase_timestamp'].dt.year
        df['order_month_str'] = df['order_purchase_timestamp'].dt.strftime('%Y-%m')
        LOGGER.info("Created order_month and order_year features")
    
    # Delivery performance categorization
    if 'is_late_delivery' in df.columns:
        df['delivery_performance'] = df['is_late_delivery'].apply(
            lambda x: 'on_time' if x == 0 else 'late' if x == 1 else 'unknown'
        )
        LOGGER.info("Created delivery_performance feature")
    
    # Review sentiment categorization by score
    if 'avg_review_score' in df.columns:
        df['review_quality'] = pd.cut(
            df['avg_review_score'],
            bins=[-0.1, 0, 2, 3, 4, 5],
            labels=['no_review', 'poor', 'fair', 'good', 'excellent']
        )
        LOGGER.info("Created review_quality feature")
    
    # Order fulfillment status
    if 'is_completed_order' in df.columns:
        df['order_fulfillment'] = df['is_completed_order'].apply(
            lambda x: 'completed' if x == 1 else 'incomplete'
        )
        LOGGER.info("Created order_fulfillment feature")
    
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean data for Tableau."""
    
    # Ensure no empty strings in categorical columns
    text_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in text_cols:
        df[col] = df[col].str.strip()
    
    # Remove any rows where critical IDs are missing
    critical_cols = ['order_id', 'customer_id', 'product_id']
    for col in critical_cols:
        if col in df.columns:
            before = len(df)
            df = df[df[col].notna()]
            after = len(df)
            if before != after:
                LOGGER.info(f"Removed {before - after} rows with missing {col}")
    
    return df


def optimize_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert repetitive string columns to categorical for memory efficiency."""
    
    # Columns that are good candidates for categorical
    categorical_candidates = [
        'order_status',
        'order_stage',
        'product_category_name',
        'product_category_name_english',
        'customer_state',
        'state_city',
        'dominant_payment_type',
        'dominant_sentiment',
        'is_expensive_item',
        'is_delivered_clean',
        'is_completed_order',
        'is_late_delivery',
        'payment_method_count',
        'delivery_performance',
        'review_quality',
        'order_fulfillment'
    ]
    
    for col in categorical_candidates:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype('category')
            LOGGER.info(f"Converted {col} to category type")
    
    return df


def main(
    project_root: Path = None,
    input_file: str = 'data/processed/master_dataset.csv',
    output_file: str = 'data/processed/tableau_ready.csv'
) -> None:
    """Main pipeline for final data preparation."""
    
    setup_logging()
    
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    
    input_path = project_root / input_file
    output_path = project_root / output_file
    
    LOGGER.info("="*60)
    LOGGER.info("Final Load Preparation for Tableau")
    LOGGER.info("="*60)
    
    # Load data
    df = load_data(input_path)
    LOGGER.info(f"Initial memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Clean and prepare
    df = remove_duplicates(df)
    df = convert_date_columns(df)
    df = handle_missing_values(df)
    df = create_derived_features(df)
    df = validate_data(df)
    df = optimize_numeric_columns(df)
    df = optimize_categorical_columns(df)
    
    # Final validation
    LOGGER.info(f"Final shape: {df.shape}")
    LOGGER.info(f"Final memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    LOGGER.info(f"Final null count: {df.isnull().sum().sum()}")
    
    # Save to CSV
    LOGGER.info(f"Saving to {output_path}")
    df.to_csv(output_path, index=False)
    
    file_size = output_path.stat().st_size / 1024**2
    LOGGER.info(f"File size: {file_size:.2f} MB")
    
    LOGGER.info("="*60)
    LOGGER.info("✓ Tableau-ready dataset created successfully!")
    LOGGER.info("="*60)


if __name__ == '__main__':
    main()
