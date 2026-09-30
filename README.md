[![Tests](https://github.com/QTTN1/IDS_706_dpf_demo/actions/workflows/tests.yml/badge.svg)](https://github.com/QTTN1/IDS_706_dpf_demo/actions/workflows/tests.yml)

My data engineering project using pandas to explore a sales dataset from Kaggle.

## Setup

```
conda create -n data706 python=3.12 pandas scipy numpy scikit-learn matplotlib seaborn -c conda-forge
conda activate data706
pip install kagglehub
python Week2_Mini_Assignment.py
```

Script downloads the dataset automatically with kagglehub, so no manual download needed.

## Dataset

[Sales Dataset (E-Commerce Sales)](https://www.kaggle.com/datasets/naofilahmad/sales-datset-product-sample) from Kaggle. ~186k orders, 8 product categories, 9 cities, 1 year (12 months) of data.

## What I did

**Import + Inspect**

Loaded the CSV with pandas, used `.head()`, `.info()`, `.describe()` to investigate, checked for missing values/duplicates (there weren't any). Dropped an extra blank column, and transformed Order Date to an actual date.

**Filtering + Grouping**

Additional dataset investigation, and setup for the ML algorithm.

I filtered things like high-value orders (Sales > $500) and orders from specific cities. Used `groupby()` to get avg price/total sales per category, and avg sale amount by time of day, and total sale by month.

**ML - Linear Regression**

Grouped sales by month per category, then fit a linear regression (Sales ~ Month) per category to see which ones are trending up. I also did the same for order count and average order value, since a category could be growing in orders without growing much in revenue, or vice versa.

Results: every category trended up overall, and followed a similar monthly trend. Laptops and Computers had the strongest revenue trend, and the value per order was also increasing. Batteries, Charging Cables, and Audio Devices had a lot more orders each month but barely moved revenue, likely since those are cheaper items.

**Visualization**

Plots showing monthly, and overall trend of sales for each category(all_category_trend.png). Scatter plot of order count trend vs. sales trend, colored by avg order value trend, so all three show up in one visualization(trend_scatter).


## Testing & CI

Second part of this project. 

I had to move the notebook's process into `sales_analysis.py` and break it down again, into functions, before I could test it.

Tests are in `tests/test_sales_analysis.py`, built one at a time with small made-up data instead of the real dataset, so they run fast and don't need my Kaggle login or internet to work. I implemented tests for the cleaning step, the city filter, and the linear regression trend calculation, plus one bigger test that runs the whole pipeline from a CSV file ,to check everything still connects correctly.

Run tests with:

```
pytest tests/ -v
```

Screenshot of my tests passing locally:

![pytest results](screenshots/pytest_results.png)

Also set up GitHub Actions so these run automatically every time I push to main (`.github/workflows/tests.yml`). Here it is passing on GitHub:

![GitHub Actions passing](screenshots/github_actions_passing.png)

## Also in this repo

(Not relevant to sales analysis portion)

`rust_vs_python_intro.ipynb`  (Rust ownership practice)

`movielens_dataframe_engines_simple.ipynb` (data engineering  walkthrough)



Note: Refractor, beacuse even good code needs a spa day 🛁 !
