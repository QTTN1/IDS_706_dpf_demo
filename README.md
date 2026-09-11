# Week 2 Data Analysis - Sales Dataset

Part 1 of my 3-week data engineering project. Using pandas to explore a sales dataset from Kaggle, do some filtering/grouping, and try out a simple ML model to look at sales trends.

## Setup

```
conda create -n data706 python=3.12 pandas scipy numpy scikit-learn matplotlib seaborn -c conda-forge
conda activate data706
pip install kagglehub
python Week2_Mini_Assignment.py
```

Script downloads the dataset automatically with kagglehub, no manual download needed.

## Dataset

[Sales Dataset (E-Commerce Sales)](https://www.kaggle.com/datasets/naofilahmad/sales-datset-product-sample) from Kaggle. ~186k orders, 8 product categories, 9 cities, 12 months of data.

## What I did

**Import + Inspect**

Loaded the CSV with pandas, used `.head()`, `.info()`, `.describe()` to check it out, checked for missing values/duplicates (there weren't any). Dropped a leftover index column, cleaned up whitespace in the City column, and converted Order Date to an actual date.

**Filtering + Grouping**

Filtered things like high-value orders (Sales > $500) and orders from specific cities. Used `groupby()` to get avg price/total sales per category, total sales per city, and avg sale amount by time of day.

**ML - Linear Regression**

Grouped sales by month per category, then fit a linear regression (Sales ~ Month) per category to see which ones are trending up. Also did the same for order count and average order value, since a category could be growing in orders without growing much in revenue, or vice versa.

Results: every category trended up overall, but not for the same reasons. Laptops and Computers had the strongest revenue trend, and the value per order was also increasing. Batteries, Charging Cables, and Audio Devices had a lot more orders each month but barely moved revenue, since those are cheap items.

**Visualization**

Scatter plot of order count trend vs. sales trend, colored by avg order value trend, so all three show up in one chart.
