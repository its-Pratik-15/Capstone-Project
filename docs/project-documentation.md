# Capstone Project Documentation

## 1. Project Title
Brazilian E-commerce Data Analysis Pipeline

## 2. Project Overview
This project is a data processing and analysis pipeline using a Brazilian e-commerce dataset. It involves extracting raw CSV data files (such as orders, products, customers, and reviews), cleaning and merging them into a unified dataset, performing exploratory and statistical data analysis in Jupyter notebooks, and outputting an optimized dataset intended for future use in Tableau.

## 3. Problem Statement
E-commerce businesses track data across multiple systems, including sales, logistics, customer feedback, and payments. The goal of this project is to consolidate this fragmented data to uncover insights regarding customer behavior, delivery performance, and overall customer satisfaction.

## 4. Project Objectives
- Merge multiple relational raw data CSV files into a single master dataset.
- Clean the data by handling missing values and removing duplicates.
- Create new features such as delivery time to aid in analysis.
- Perform Exploratory Data Analysis (EDA) to identify business trends.
- Conduct statistical analysis and basic regression modeling to explore relationships in the data.
- Produce a clean dataset prepared for Tableau visualization.

## 5. Dataset / Data Source
- **Data Source**: Olist E-commerce dataset (Brazilian e-commerce).
- **Data Type**: Relational CSV files.
- **Important Columns**: `order_id`, `customer_id`, `product_id`, `price`, `freight_value`, `payment_value`, `review_score`, `order_purchase_timestamp`, `order_delivered_customer_date`.
- **Dataset Size**: The 9 raw CSV files contain approximately 1.6 million total rows. The final merged master dataset contains 112,651 records.
- **Limitations**: The dataset contains some missing values (e.g., missing product dimensions and unwritten review comments).

## 6. Data Cleaning and Preprocessing
The data cleaning process is implemented in Python scripts (`scripts/etl_pipeline.py`, `scripts/final_load_prep.py`) and Jupyter notebooks (`notebooks/02-cleaning.ipynb`). Key steps included:
- **Missing Values**: Missing text values were filled with "unknown", "no_title", or "no_comment". Numeric values like product dimensions were filled using the median.
- **Duplicate Records**: Duplicate rows were removed using the pandas `drop_duplicates()` function.
- **Inconsistent Values**: Text columns were standardized by converting to lowercase and stripping extra spaces.
- **Data Type Conversion**: Timestamp columns were converted to datetime objects. Numeric columns were optimized for memory usage (e.g., converting to `int32` or `float32`).
- **Feature Engineering**: New columns were created, such as:
  - `delivery_time_days`: The difference between the delivery date and the purchase date.
  - `is_late_delivery`: A flag indicating if the actual delivery date exceeded the estimated delivery date.
  - `sentiment`: Categorized from the `review_score`.

## 7. Exploratory Data Analysis
Exploratory Data Analysis was conducted in `notebooks/03-eda.ipynb` and covered:
- **Univariate & Bivariate Analysis**: Examined distributions of prices, freight values, and review scores.
- **Time Series Analysis**: Looked at monthly trends in order counts, total revenue, and average order value. 
- **Geographic Analysis**: Checked top states by order volume and average delivery times.
- **Product Analysis**: Grouped data to find the top product categories by order counts and total revenue.
- **Insights Found**: Explored relationships such as how delivery speed correlates with review scores, and identified top-performing categories and states.

## 8. Data Visualization
Visualizations were created during the EDA and statistical analysis phases using Python libraries:
- **Charts Created**: Bar charts for top product categories and geographic distributions, line charts for time series trends, and histograms for statistical distributions.
- **Tools Used**: Matplotlib and Seaborn in Jupyter Notebooks.
- *Note: While a CSV file is exported to the `tableau/` folder, no actual Tableau dashboard files (.twbx) are present in the repository.*

## 9. Statistical Analysis / Model / Business Logic
Statistical analysis was performed in `notebooks/04-statistical_analysis.ipynb` using `scipy` and `scikit-learn`:
- **Models Used**: 
  - **Linear Regression**: Explored relationships to predict continuous review scores.
  - **Logistic Regression**: Used to classify high customer satisfaction.
- **Input/Output**: Inputs included numerical features like delivery time and freight value. Outputs were predicted review scores or satisfaction probability.
- **Evaluation Metrics**: Used R-squared and Mean Squared Error for Linear Regression, and a Classification Report for Logistic Regression.
- **Statistical Tests**: Included the Mann-Whitney U Test and Chi-Square tests to evaluate distributions and categorical relationships.

## 10. Project Architecture
The repository is structured as follows:
- `data/`: Contains `raw/` (original CSV files) and `processed/` (cleaned datasets like `master_dataset.csv`).
- `docs/`: Holds project documentation.
- `notebooks/`: Contains step-by-step Jupyter notebooks for extraction (`01`), cleaning (`02`), EDA (`03`), statistical analysis (`04`), and final prep (`05`).
- `scripts/`: Python scripts (`etl_pipeline.py`, `final_load_prep.py`) for running data transformations.
- `tableau/`: Stores the final prepared CSV dataset.

## 11. Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook
- GitHub

## 12. Team Contributions
Based on the visible repository files and available project structure, the team contribution appears to be:

| Team Member | Contribution |
|------------|--------------|
| Ayush Kumar Jha | Project folder structure, comprehensive EDA, statistical analysis, final load prep notebook, and code reviews. |
| Pratik Kumar Pan | Added raw data files, data extraction, and initial cleaning notebooks. |
| Harsh Patel | Developed the Python ETL pipeline script, final load prep script, and generated the Tableau-ready CSV. |
| Adarsh Priydarshi | Repository maintenance and `.gitignore` configurations. |
| Saubhagya Anubhav | Comprehensive project documentation and repository analysis. |

*Exact individual contributions should be verified from GitHub commit history before final submission.*

## 13. Challenges Faced
Based on the code and data structure, challenges likely included:
- Joining 9 different datasets using varying primary and foreign keys (`order_id`, `product_id`, `customer_id`).
- Handling missing product dimensions and unwritten review comments.
- Managing memory usage for the master dataset, which was addressed by optimizing data types in the final preparation script.

## 14. Final Output / Results
The final project produces:
- A cleaned and joined `master_dataset.csv` containing order, product, customer, and review data.
- A `tableau_ready.csv` dataset optimized for memory and speed.
- Analysis reports available within the Jupyter notebooks.
- Python scripts that can execute the data transformation steps.

## 15. How to Run the Project
To run the data processing scripts locally:

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. Run the ETL pipeline script to process the raw data:
```bash
python scripts/etl_pipeline.py
```

3. Run the final preparation script to generate the optimized dataset:
```bash
python scripts/final_load_prep.py
```

4. To view the analysis, open the notebooks sequentially using Jupyter:
```bash
jupyter notebook
```
Navigate to the `notebooks/` folder and explore files `01` through `05`.

## 16. Future Improvements
- Develop the actual Tableau dashboard files to visualize the prepared data.
- Improve model performance by testing additional features and comparing different algorithms.
- Set up a formal data pipeline tool (like Apache Airflow) if the project needs to run on a daily schedule.

## 17. Conclusion
This project demonstrates how raw e-commerce data can be cleaned, merged, analyzed, and prepared for visualization. The work highlights important areas such as customer behavior, delivery performance, and review patterns. Overall, the project shows how data analysis can support better business understanding and decision-making.

## 18. Documentation Accuracy Notes
- **Tableau Visualizations**: The repository contains a folder named `tableau/` and code that exports a "Tableau-ready CSV", but it does not contain any actual Tableau dashboard files or screenshots. Visualizations were confirmed to be done in Python (Matplotlib/Seaborn).
- **Automation**: The documentation refers to "automated Python scripts" because the `etl_pipeline.py` script can run the transformations in one command, but there is no evidence of scheduled automation (e.g., cron jobs or Airflow) in the repository.
