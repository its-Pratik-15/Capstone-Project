# Project Report: Brazilian E-commerce Data Analysis Pipeline

---

## 1. Title Page

**Project Title:** Brazilian E-commerce Data Analysis Pipeline  
**Authors:** Ayush Kumar Jha, Pratik Kumar Pan, Harsh Patel, Adarsh Priydarshi, Saubhagya Anubhav  
**Date:** April 29, 2026  
**Organization:** Capstone Project Team  

---

## 2. Abstract

The explosive growth of e-commerce necessitates robust data pipelines to extract actionable insights from vast, disparate datasets. This project focuses on the end-to-end data processing and exploratory data analysis (EDA) of the Olist Brazilian E-commerce dataset, which encompasses over 100,000 orders. Our objective was to consolidate multi-dimensional data spanning customer demographics, product characteristics, payments, logistics, and customer reviews into a unified architecture. By engineering a scalable ETL (Extract, Transform, Load) pipeline, we cleansed and harmonized 9 relational datasets, addressing data inconsistencies, missing values, and high dimensionality. 

Subsequent exploratory analyses revealed critical business patterns, notably the direct correlation between delivery efficiency and customer satisfaction. The pipeline culminates in an optimized, memory-efficient dataset structured for seamless integration with business intelligence tools like Tableau. Ultimately, this project delivers a foundational data infrastructure and an interactive dashboard providing actionable analytical insights to drive data-driven decision-making in logistics and customer experience management.

---

## 3. Introduction

### Background and Motivation
Modern e-commerce platforms generate massive volumes of transactional and behavioral data. While this data holds immense potential, its fragmented nature across various systems (sales, logistics, payments, feedback) often impedes comprehensive analysis. The Olist dataset provides a realistic representation of this challenge, offering a rich, multi-faceted view of Brazilian e-commerce operations. Understanding the underlying drivers of customer satisfaction and operational bottlenecks is crucial for maintaining a competitive edge.

### Problem Statement
E-commerce businesses struggle to consolidate fragmented data silos to uncover meaningful insights regarding customer behavior, delivery performance, and overall satisfaction. Without a unified view, identifying inefficiencies in the supply chain or understanding the root causes of poor customer feedback becomes a daunting task.

### Objectives
- **Data Consolidation:** Merge 9 distinct relational raw CSV files into a single, cohesive master dataset.
- **Data Cleansing:** Systematically handle missing values, anomalies, and duplicates to ensure data integrity.
- **Feature Engineering:** Develop new, impactful metrics such as delivery time and review sentiment to facilitate deeper analysis.
- **Exploratory Data Analysis (EDA):** Identify key business trends related to geography, product performance, and time series patterns.
- **BI Readiness & Visualization:** Generate a highly optimized dataset and deploy an interactive dashboard tailored for business intelligence analysis.

---

## 4. Project Scope

### In-Scope
- Extraction and loading of raw Olist CSV datasets.
- Data cleaning, normalization, and handling of missing values across all tables.
- Feature engineering for delivery metrics, order values, and review sentiments.
- Comprehensive EDA using Python visualization libraries (Matplotlib, Seaborn).
- Exporting a final, memory-optimized dataset for downstream BI consumption.
- Creation of an interactive Tableau dashboard to monitor key performance indicators (KPIs).

### Out-of-Scope
- Real-time data streaming or active API integration (the project relies on a static historical dataset).
- Automated deployment to Tableau Server/Online (the interactive `.twbx` dashboard is built and available locally, but not deployed to a server).

### Key Assumptions
- The provided Olist dataset accurately reflects real-world operational metrics without synthetic skew.
- Missing delivery dates primarily indicate orders that are still in transit or were canceled, rather than systemic data loss.
- Review scores are the primary and most accurate proxy for customer satisfaction.

---

## 5. System Architecture / Workflow

The system architecture is designed as a sequential, batch-processing ETL pipeline transitioning into analytical workflows.

### End-to-End Pipeline Explanation
1. **Extraction:** Raw CSV files are read into Pandas DataFrames.
2. **Transformation (Cleaning & Merging):** 
   - Initial data cleaning and iterative exploration were conducted in `notebooks/02-cleaning.ipynb`.
   - To ensure reproducibility and automation (as running notebooks in production is not ideal), the cleaning logic was extracted and modularized into Python scripts (`scripts/etl_pipeline.py`).
   - Standardization of text strings, datetime parsing, and handling of nulls via median imputation or placeholder strings.
   - Merging relational tables using primary/foreign keys (`order_id`, `product_id`, `customer_id`).
3. **Analytical Processing:** 
   - Execution of Jupyter Notebooks (`03-eda.ipynb`) for trend discovery.
4. **Final Load Prep:** 
   - Execution of `scripts/final_load_prep.py` to downcast data types for memory optimization, producing `tableau_ready.csv`.

### Data Flow Diagram (Textual Representation)
```text
[Raw CSV Data (data/raw/)]
       |
       v
[ETL Script (scripts/etl_pipeline.py)] ---> Standardizes, cleans, and merges data
       |
       v
[Processed Data (data/processed/master_dataset.csv)]
       |
       +---> [EDA & Visualization (notebooks/03-eda.ipynb)]
       |
       v
[Final Load Prep (scripts/final_load_prep.py)] ---> Downcasts data types, final filtering
       |
       v
[BI-Ready Dataset (data/processed/tableau_ready.csv)] ---> [Tableau Dashboard]
```

---

## 6. Technology Stack

- **Language:** Python 3 (Chosen for its robust data manipulation ecosystem).
- **Data Manipulation:** Pandas, NumPy (Essential for memory-efficient handling of large, relational tabular data).
- **Visualization:** Matplotlib, Seaborn (Used for generating static, high-quality analytical charts in notebooks).
- **Environment:** Jupyter Notebooks (Ideal for iterative EDA and narrative-driven analysis), standard Python scripts for production pipeline tasks.
- **BI Tool:** Tableau (Employed for developing the interactive E-commerce Performance & Delivery Insights Dashboard).

---

## 7. Data Description

### Dataset Source
The dataset is the Brazilian E-Commerce Public Dataset by Olist, originally sourced via Kaggle. It contains information on 100k orders made at multiple marketplaces in Brazil.

### Schema and Features
The data operates on a relational schema centered around orders:
- **Orders:** `order_id`, `customer_id`, `order_status`, timestamps.
- **Items:** `product_id`, `seller_id`, `price`, `freight_value`.
- **Products:** Categories, physical dimensions, weight.
- **Customers/Sellers:** Geolocation data (city, state, zip).
- **Payments:** Payment type, installments, payment value.
- **Reviews:** Review score (1-5), comments.

### Data Challenges
- **Missing Values:** Extensive missing text in review comments; missing physical dimensions for certain products; missing delivery dates for in-transit orders.
- **High Dimensionality & Memory:** Merging 9 tables results in a highly denormalized dataset that consumes significant RAM, necessitating strict data type management.
- **Complex Relationships:** Ensuring one-to-many relationships (e.g., one order to multiple items/payments) do not unintentionally inflate revenue calculations via row duplication.

---

## 8. Data Preprocessing & Cleaning

*Note: All data preprocessing logic was iteratively developed and tested within `notebooks/02-cleaning.ipynb`. Once finalized, this logic was encapsulated into `scripts/etl_pipeline.py` and `scripts/final_load_prep.py` to allow for scalable execution, since running Jupyter Notebooks directly is not a best practice for automated pipelines.*

### Handling Null Values
- **Categorical/Text:** Missing product categories were filled with `"unknown"`. Missing review titles and messages were filled with `"no_title"` and `"no_comment"`.
- **Numerical/Dimensions:** Missing product dimensions and weights were imputed using the median values of their respective columns to prevent skewing distributions.
- **Dates:** Missing actual delivery dates were retained as nulls for in-transit orders, and corresponding delivery time calculations were set to `0` safely.

### Feature Engineering
- `delivery_time_days`: The temporal difference between `order_delivered_customer_date` and `order_purchase_timestamp`.
- `is_late_delivery`: Boolean flag comparing actual delivery against `order_estimated_delivery_date`.
- `sentiment`: Categorization of numeric `review_score` into actionable labels (e.g., Positive, Neutral, Negative).
- `item_total_value`: The sum of `price` and `freight_value`.

### Transformations Applied
- **Standardization:** All string columns (cities, states, categories) were converted to lowercase and stripped of leading/trailing whitespace.
- **Data Type Optimization:** Safely downcasted `float64` to `float32`, integers to `int32`, and converted repetitive strings into `category` types to drastically reduce the final CSV footprint in `scripts/final_load_prep.py`.
- **Deduplication:** Applied `.drop_duplicates()` rigorously post-merge to prevent double-counting.

---

## 9. Exploratory Data Analysis (EDA)

The EDA phase uncovered several actionable insights, visualized using Python libraries (artifacts saved in `reports/`):

- **Geographic Disparities:** A heavy concentration of orders originates from the SP (São Paulo) state, which also enjoys the fastest average delivery times. Remote states like RR (Roraima) suffer from significantly longer delivery windows and higher freight costs.
- **Category Performance:** "Bed_bath_table" and "health_beauty" emerged as top-performing categories by sheer order volume, while "watches_gifts" drove significant gross revenue due to higher price points.
- **Time Series Trends:** Revenue and order volume demonstrated distinct seasonal peaks, likely corresponding to major e-commerce events (e.g., Black Friday) in the Brazilian market.
- **Freight vs. Price:** Analysis revealed that for lower-priced items, freight costs often represent a disproportionately high percentage of the total customer cost, which negatively impacts conversion and satisfaction.

---

## 10. Dashboard / Visualization

The project includes an interactive BI dashboard built in Tableau. The actual dashboard file (`DVA Capstone2 Dashboard.twbx`) is located in the `tableau/file/` directory, and a visual screenshot of the dashboard is available in `tableau/screenshot/`.

**E-commerce Performance & Delivery Insights Dashboard:**
Based on our findings, we designed a comprehensive dashboard that tracks high-level performance and drills down into operational efficiency and customer satisfaction. The dashboard components include:
- **Core KPIs:** Provides an immediate view of business health, tracking **Total Revenue ($13.5M+)**, **Total Orders (81K+)**, an **Order Completion Rate (97.08%)**, and an **Average Delivery Time of 12.02 Days**.
- **Interactive Filters:** Global filters allow users to slice data dynamically by **Order Purchase Timestamp** and **Order Stage** (e.g., filtering strictly for 'Approved' orders).
- **Revenue Trend:** A line chart displaying revenue over time, highlighting a distinct peak in May and a sharp decline in September, pointing to seasonality in purchasing behavior.
- **Payment Type Distribution:** A horizontal bar chart showing that **Card Payments** overwhelmingly dominate the payment methods, followed by Cash/Bank Transfers and Vouchers.
- **Order Stage Distribution:** Visually confirms that the vast majority of tracked records have successfully reached the "Delivered" state.
- **Customer Experience (Sentiment):** Analyzes review data, showcasing that **positive** sentiment strongly outweighs negative and neutral feedback, giving a clear indication of overall platform health.
- **Delivery Delay Analysis:** A bar chart tracking the ratio of "Delayed" versus "On Time" shipments, providing logistics teams with an instant read on shipping performance.

---

## 11. Key Decisions & Trade-offs

- **Decision: Median Imputation for Product Dimensions.** 
  - *Trade-off:* While preserving row counts, it artificially reduces variance in weight/size metrics. *Alternative Considered:* Dropping rows. *Justification:* Dropping rows would mean losing valuable transaction data linked to those products.
- **Decision: Categorizing Sentiment.**
  - *Trade-off:* Condensing 1-5 scores into "Positive/Negative" loses granular nuance but dramatically simplifies the classification objective for business reporting.
- **Decision: Pre-computing BI Metrics in Python.**
  - *Trade-off:* Increases the width of the final CSV but significantly reduces the processing load on Tableau, optimizing dashboard load times and user experience.

---

## 12. Results & Outcomes

- **Technical Impact:** Successfully engineered a reproducible pipeline that reduces 1.6 million raw rows across 9 files into a streamlined, memory-optimized ~112,000 row analytical dataset ready for immediate BI ingestion.
- **Business Impact:** Delivered a comprehensive dashboard that empirically visualizes the relationships between revenue, payment preferences, logistics performance, and customer sentiment. Logistics teams can now use the dashboard to set scientifically backed SLAs (Service Level Agreements) for regional shipping partners based on historical delay statuses.

---

## 13. Challenges Faced

1. **Relational Complexity:** Joining 9 tables created a high risk of Cartesian explosions (row duplication). 
   *Solution:* Implemented strict left joins and continuous deduplication steps throughout `etl_pipeline.py`.
2. **Memory Constraints:** The fully denormalized DataFrame initially exceeded standard memory limits on local machines.
   *Solution:* Implemented a rigorous data-type downcasting function in `final_load_prep.py` (e.g., `float64` to `float32`, object to categorical).
3. **Inconsistent Text Data:** Categorical variables like city names had varied spellings and casing.
   *Solution:* Applied robust string normalization techniques (lowercasing, stripping whitespace) prior to aggregation.

---

## 14. Future Improvements

- **Scalability:** Migrate the local Python scripts to a managed orchestration tool like Apache Airflow or Prefect to handle scheduled, incremental daily loads instead of full historical batch processing.
- **Feature Enhancements:** Integrate NLP sentiment analysis on the `review_comment_message` column using tools like VADER or HuggingFace transformers to extract deeper nuance beyond the numeric score.
- **BI Deployment:** Publish the locally developed Tableau interactive dashboard (`DVA Capstone2 Dashboard.twbx`) to Tableau Public or Tableau Server for broader stakeholder access.

---

## 15. Conclusion

This project successfully demonstrates the transformation of raw, fragmented e-commerce data into a cohesive, analytical powerhouse. By constructing a robust ETL pipeline and performing rigorous EDA, we uncovered the critical intersection of logistics performance and customer satisfaction. The structured approach to memory optimization and feature engineering ensured that the final dataset was primed for enterprise-grade BI visualization. The resulting Tableau dashboard lays the groundwork for data-driven logistical improvements and enhanced customer experiences.

---

## 16. References

1. **Dataset:** Olist Brazilian E-Commerce Public Dataset (Kaggle).
2. **Libraries:** 
   - Pandas Documentation: https://pandas.pydata.org/
   - Matplotlib & Seaborn for Python Visualizations.
3. **Methodologies:** Standard ETL architecture and data visualization best practices.
