# Data Documentation

## 1. Data Overview
- **Dataset name**: Olist E-Commerce Dataset (Brazilian E-Commerce Public Dataset)
- **Data source**: Olist (Public dataset originally provided on Kaggle)
- **Type of data**: Relational database tables provided as CSV files, containing historical e-commerce order data from Olist, a Brazilian marketplace platform.
- **Purpose of using this dataset in the project**: To analyze e-commerce performance, understand customer satisfaction, evaluate delivery efficiency, and prepare a clean, unified dataset for reporting and visualization.

## 2. Raw Data Files
| File Name | Purpose | Important Columns | Notes |
|----------|---------|------------------|-------|
| `olist_orders_dataset.csv` | Core table containing order level details and timestamps. | `order_id`, `customer_id`, `order_status`, `order_purchase_timestamp`, `order_delivered_customer_date` | Connects to customers, items, payments, and reviews. |
| `olist_order_items_dataset.csv` | Contains details of individual items within an order. | `order_id`, `order_item_id`, `product_id`, `seller_id`, `price`, `freight_value` | Connects orders with products and sellers. |
| `olist_products_dataset.csv` | Contains product characteristics. | `product_id`, `product_category_name`, `product_weight_g`, dimensions | Connects to items and translation tables. |
| `olist_customers_dataset.csv` | Contains customer geographical details. | `customer_id`, `customer_unique_id`, `customer_city`, `customer_state` | Connects to orders. |
| `olist_order_payments_dataset.csv` | Contains payment method and value details. | `order_id`, `payment_type`, `payment_installments`, `payment_value` | Connects to orders. |
| `olist_order_reviews_dataset.csv` | Contains customer review ratings and comments. | `review_id`, `order_id`, `review_score`, `review_comment_title`, `review_comment_message` | Connects to orders. |
| `product_category_name_translation.csv` | Translates category names from Portuguese to English. | `product_category_name`, `product_category_name_english` | Connects to products. |
| `olist_sellers_dataset.csv` | Contains seller geographical details. | `seller_id`, `seller_city`, `seller_state` | Connects to items. |
| `olist_geolocation_dataset.csv` | Contains mapping of zip codes to lat/long coordinates. | `geolocation_zip_code_prefix`, `geolocation_lat`, `geolocation_lng`, `geolocation_city` | Available for location-based analysis, but not clearly used in the current pipeline. |

## 3. Data Relationships
The data is structured as a relational database around orders.
- **Orders connect with customers** using `customer_id`.
- **Orders connect with items** using `order_id`.
- **Orders connect with payments** using `order_id`.
- **Orders connect with reviews** using `order_id`.
- **Items connect with products** using `product_id`.
- **Items connect with sellers** using `seller_id`.
- **Products connect with translation** using `product_category_name`.

## 4. Important Columns
| Column Name | Meaning | Used In | Notes |
|------------|---------|---------|-------|
| `order_id` | Unique identifier for an order | orders, items, payments, reviews | Primary key connecting most tables |
| `customer_id` | Unique identifier for a customer per order | orders, customers | Keys to customer demographic data |
| `product_id` | Unique identifier for a product | items, products | Keys to product characteristics |
| `seller_id` | Unique identifier for a seller | items, sellers | Keys to seller location |
| `price` | Price of an individual item | items | Used for total value calculations |
| `freight_value` | Shipping cost for the item | items | Used for total value calculations |
| `order_purchase_timestamp` | When the order was placed | orders | Baseline date for delivery times |
| `order_delivered_customer_date` | When the order reached the customer | orders | Used to compute actual delivery time |
| `order_estimated_delivery_date` | When the order was expected | orders | Used to flag late deliveries |
| `review_score` | Rating from 1 to 5 | reviews | Used for sentiment analysis |
| `payment_value` | Total amount paid by customer | payments | Used to calculate order revenue |
| `product_category_name_english` | English name of product category | translation, master | Used for product groupings |

## 5. Data Cleaning Process
The cleaning process is managed by `scripts/etl_pipeline.py` and `scripts/final_load_prep.py`.

- **Standardizing text**: Text columns (like city, state, product category, payment type) were stripped of whitespace and converted to lowercase. Done in `etl_pipeline.py`.
- **Converting date columns**: Timestamps across orders and reviews were parsed into explicit datetime objects. Done in `etl_pipeline.py` and `final_load_prep.py`.
- **Handling incomplete delivery dates**: Computed columns like `delivery_time_days` only where dates were valid. Filled missing calculations with 0 and missing dates with nulls. Done in `final_load_prep.py`.
- **Cleaning review comments**: Filled missing review titles with `no_title` and missing messages with `no_comment`. Done in `etl_pipeline.py`.
- **Cleaning product dimensions**: Replaced missing numeric values with the median value of the column. Done in `etl_pipeline.py`.
- **Removing duplicates**: Removed duplicate rows on all tables before writing outputs. Done in `etl_pipeline.py` and `final_load_prep.py`.

## 6. Missing Value Handling
| Column / Field | Missing Value Issue | Handling Method | Source File / Script |
|---------------|--------------------|-----------------|----------------------|
| `product_category_name` | Missing categories | Filled with `"unknown"` | `scripts/etl_pipeline.py` |
| Product dimensions / weight | Missing numeric metrics | Filled with median | `scripts/etl_pipeline.py` |
| `review_comment_title` | Reviews without titles | Filled with `"no_title"` | `scripts/etl_pipeline.py` |
| `review_comment_message` | Reviews without text | Filled with `"no_comment"` | `scripts/etl_pipeline.py` |
| Delivery time metrics | Missing delivery dates | Filled with `0` | `scripts/final_load_prep.py` |
| Missing string columns | Miscellaneous text | Filled with `"unknown"` | `scripts/final_load_prep.py` |
| Summary payment/review metrics | Missing grouped stats | Filled with `0` or `"no_review"` | `scripts/final_load_prep.py` |

## 7. Duplicate Handling
- **Files checked**: All core files and the merged `master_dataset`.
- **How they were removed**: Duplicate rows were completely dropped.
- **Method used**: The `.drop_duplicates()` function in Python.
- **Why it was important**: Essential to prevent double-counting, which artificially inflates revenue, order counts, or review scores in visualizations.

## 8. Data Type Conversion
- **Date Columns**: Timestamp columns were converted to datetime formats.
- **Numeric Optimization**: Memory optimization was performed by safely downcasting float columns to nullable integers or `float32`, and integers to `int32`. Done in `scripts/final_load_prep.py`.
- **Categorical Optimization**: Columns with repetitive text strings were converted to categorical data types to reduce file size. Done in `scripts/final_load_prep.py`.

## 9. Feature Engineering
| New Feature | Created From | Purpose |
|------------|--------------|---------|
| `delivery_time_days` | `order_delivered_customer_date`, `order_purchase_timestamp` | To measure how fast orders are delivered |
| `is_late_delivery` | `order_delivered_customer_date`, `order_estimated_delivery_date` | To flag if an order missed its expected date |
| `sentiment` | `review_score` | To categorize ratings into positive, neutral, and negative |
| `item_total_value` | `price`, `freight_value` | To find the full cost of an individual item |
| `product_volume_cm3` | length, height, width | To analyze shipping volume requirements |
| `payment_value_per_installment` | `payment_value`, `payment_installments` | To see average amount paid per billing cycle |
| `order_total_with_freight` | `order_items_total_value`, `order_items_total_freight` | Sum total value of the entire order |
| `shipping_cost_percentage` | freight, order value | To analyze the burden of shipping costs |
| `order_complexity` | item count, distinct sellers | To identify multi-item, multi-seller logistical difficulty |
| `delivery_performance` | `is_late_delivery` | Categorical label marking order as "on_time" or "late" |
| `review_quality` | `avg_review_score` | Binning scores into poor, fair, good, excellent |
| `order_month_str` | `order_purchase_timestamp` | Formats date into YYYY-MM for time series grouping |

## 10. Final Processed Data
Final processed files are written to `data/processed/`.
- **Cleaned Base Tables**: `orders_clean.csv`, `items_clean.csv`, `products_clean.csv`, `customers_clean.csv`, `payments_clean.csv`, `reviews_clean.csv`, `translation_clean.csv`, `sellers_clean.csv`, and `geolocation_clean.csv`.
- **`master_dataset.csv`**: A large, denormalized table joined at the order level. Used for exploratory data analysis.
- **`tableau_ready.csv`**: An optimized and further cleaned version of `master_dataset.csv`.
- **Tableau Assets**: The `tableau/` folder currently only contains a `.gitignore` file. There is no Tableau dashboard (`.twb` or `.twbx`) in the repository at this time.

## 11. Data Quality Notes
- **Missing Delivery Dates**: Some orders are not yet delivered, resulting in empty delivery date fields.
- **Missing Product Dimensions**: Some items lacked basic measurements, which were imputed using medians.
- **Missing Review Comments**: The vast majority of reviews lack written comments.
- **Memory/Size Considerations**: The merged dataset can become large, so type conversion and downcasting help improve memory efficiency.

## 12. Data Flow
Raw CSV files  
↓  
Cleaning and preprocessing (`scripts/etl_pipeline.py`)  
↓  
Merged master dataset (`data/processed/master_dataset.csv`)  
↓  
EDA and statistical analysis (Jupyter notebooks)  
↓  
Final Tableau-ready dataset (`scripts/final_load_prep.py` → `data/processed/tableau_ready.csv`)

## 13. How to Recreate the Data Output
Run the following commands in order from the project root:

```bash
pip install -r requirements.txt
python scripts/etl_pipeline.py
python scripts/final_load_prep.py
```

## 14. Important Notes for Contributors
- Do not directly edit files inside `data/raw/`.
- Keep cleaned files inside `data/processed/`.
- If new columns are created, document them in this file.
- If cleaning logic changes, update this documentation.
- Check missing values before analysis.
- Keep large generated files organized.
- Avoid committing unnecessary temporary files.

## 15. Summary
The data documentation details the transformation of raw Olist e-commerce CSV files into clean, analysis-ready datasets. The pipeline handles missing values, standardizes formatting, converts data types for memory optimization, and generates new features related to delivery performance, order value, and review sentiment. The final output is a consolidated, optimized dataset ready for deeper exploratory analysis and future visualization.
