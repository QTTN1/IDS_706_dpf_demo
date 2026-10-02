"""
Sales data analysis for the IDS 706 project, using a Kaggle e-commerce sales dataset.

Refactored out of Week2_Mini_Assignment.ipynb into testable functions.
"""

import os

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, Docker); must run before pyplot import

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from sklearn.linear_model import LinearRegression

# Format of "Order Date" in the raw Kaggle CSV (day-month-year hour:minute)
DATE_FORMAT = "%d-%m-%Y %H:%M"


# ---------------------------------------------------------------------------
# Loading, cleaning, data quality
# ---------------------------------------------------------------------------


def load_data(csv_path=None):
    """Load the sales dataset into a DataFrame.

    If csv_path is None, downloads the public dataset via kagglehub
    (needs network access, but no Kaggle login).
    """
    if csv_path is None:
        import kagglehub

        path = kagglehub.dataset_download("naofilahmad/sales-datset-product-sample")
        csv_path = path + "/Sales Data.csv"
    return pd.read_csv(csv_path)


def data_quality_report(raw_df):
    """Summarize data-quality issues in the *raw* data, before cleaning.

    Returns a dict with:
    - rows: number of rows
    - missing_values: total missing cells (and missing_by_column for any > 0)
    - duplicate_rows: rows that exactly repeat another row (ignoring the CSV's
      index column, which would otherwise make every row look unique)
    - sales_mismatch: rows where Sales != Quantity Ordered * Price Each
    - outliers_by_category: Sales outliers per category (1.5 * IQR rule). Computed
      per category because a $1,700 laptop is normal, while a $1,700 battery is not.
    """
    df = raw_df.drop(columns=["Unnamed: 0"], errors="ignore")
    missing = df.isna().sum()

    expected_sales = df["Quantity Ordered"] * df["Price Each"]
    sales_mismatch = int(((df["Sales"] - expected_sales).abs() > 0.01).sum())

    outliers = {}
    for category, sales in df.groupby("Product Category")["Sales"]:
        q1, q3 = sales.quantile([0.25, 0.75])
        iqr = q3 - q1
        is_outlier = (sales < q1 - 1.5 * iqr) | (sales > q3 + 1.5 * iqr)
        outliers[category] = int(is_outlier.sum())

    return {
        "rows": len(df),
        "missing_values": int(missing.sum()),
        "missing_by_column": {col: int(n) for col, n in missing.items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "sales_mismatch": sales_mismatch,
        "outliers_by_category": outliers,
    }


def clean_data(df):
    """Clean the raw sales DataFrame.

    - Drops the leftover index column from the CSV
    - Drops exact duplicate rows (the same line item recorded twice)
    - Parses "Order Date" into a datetime (day-first format)
    - Strips whitespace from "City" names
    - Fixes known typos in "Product Category" (e.g. "Batterie" -> "Batteries")

    Outliers are intentionally kept: they are real high-quantity purchases,
    not data errors (see data_quality_report).
    """
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.drop_duplicates().reset_index(drop=True)
    # Raw dates are day-first, e.g. "30-12-2019 00:01". An explicit format avoids
    # pandas guessing per row (slow + a UserWarning) and stops a date like
    # "05-12-2019" from being misread as May 12 instead of 5 December.
    df["Order Date"] = pd.to_datetime(df["Order Date"], format=DATE_FORMAT)
    df["City"] = df["City"].str.strip()
    # the raw data misspells one category as "Batterie"
    df["Product Category"] = (
        df["Product Category"].str.strip().replace({"Batterie": "Batteries"})
    )
    return df


# ---------------------------------------------------------------------------
# Filtering and summaries
# ---------------------------------------------------------------------------


def filter_high_value_orders(df, threshold=500):
    """Return only orders with Sales strictly greater than threshold."""
    return df[df["Sales"] > threshold]


def filter_by_city(df, city):
    """Return only orders from the given city."""
    return df[df["City"].str.strip() == city.strip()]


def category_summary(df):
    """Average price, total quantity, total sales, and line-item count per category."""
    return (
        df.groupby("Product Category")
        .agg(
            avg_price=("Price Each", "mean"),
            total_quantity=("Quantity Ordered", "sum"),
            total_sales=("Sales", "sum"),
            order_count=("Order ID", "count"),
        )
        .sort_values("total_sales", ascending=False)
    )


# ---------------------------------------------------------------------------
# Trend models
# ---------------------------------------------------------------------------


def _monthly_by_category(df):
    """Aggregate to one row per (category, month): total sales, order count, AOV."""
    monthly = (
        df.groupby(["Product Category", "Month"])
        .agg(total_sales=("Sales", "sum"), order_count=("Sales", "size"))
        .reset_index()
        .sort_values(["Product Category", "Month"])
    )
    monthly["avg_order_value"] = monthly["total_sales"] / monthly["order_count"]
    return monthly


def _fit_line(months, values):
    """Fit values ~ month with linear regression.

    Returns (slope, r_squared, fitted_values). With fewer than 2 points a line
    can't be fit, so slope is 0 and R^2 is NaN.
    """
    X = pd.DataFrame({"Month": months})
    model = LinearRegression().fit(X, values)
    r_squared = model.score(X, values) if len(X) >= 2 else float("nan")
    return model.coef_[0], r_squared, model.predict(X)


def monthly_category_trends(df):
    """Fit a linear regression (Sales ~ Month) per category.

    Returns one row per category with its monthly trend slope ($/month),
    sorted fastest-growing to slowest-growing.
    """
    monthly = _monthly_by_category(df)
    results = [
        {
            "Product Category": category,
            "monthly_trend_slope": _fit_line(group["Month"], group["total_sales"])[0],
        }
        for category, group in monthly.groupby("Product Category")
    ]
    # columns= keeps the expected schema even when results is empty (no data)
    return (
        pd.DataFrame(results, columns=["Product Category", "monthly_trend_slope"])
        .sort_values("monthly_trend_slope", ascending=False)
        .reset_index(drop=True)
    )


def monthly_category_trends_extended(df):
    """Like monthly_category_trends, plus order-count and average-order-value trends."""
    monthly = _monthly_by_category(df)
    results = []
    for category, group in monthly.groupby("Product Category"):
        months = group["Month"]
        results.append(
            {
                "Product Category": category,
                "sales_trend": _fit_line(months, group["total_sales"])[0],
                "order_count_trend": _fit_line(months, group["order_count"])[0],
                "avg_order_value_trend": _fit_line(months, group["avg_order_value"])[0],
            }
        )
    columns = [
        "Product Category",
        "sales_trend",
        "order_count_trend",
        "avg_order_value_trend",
    ]
    return (
        pd.DataFrame(results, columns=columns)
        .sort_values("sales_trend", ascending=False)
        .reset_index(drop=True)
    )


def seasonality_check(df, exclude_months=(12,)):
    """Is each category's growth real, or just the holiday spike?

    Refits the Sales ~ Month trend with and without the holiday month(s)
    (December by default) and reports R^2, so a reader can see how much of the
    "growth" December explains.
    """
    monthly = _monthly_by_category(df)
    results = []
    for category, group in monthly.groupby("Product Category"):
        slope_all, r2_all, _ = _fit_line(group["Month"], group["total_sales"])
        rest = group[~group["Month"].isin(exclude_months)]
        slope_ex, r2_ex, _ = _fit_line(rest["Month"], rest["total_sales"])
        results.append(
            {
                "Product Category": category,
                "slope_all_months": slope_all,
                "r2_all_months": r2_all,
                "slope_excl_holiday": slope_ex,
                "r2_excl_holiday": r2_ex,
            }
        )
    columns = [
        "Product Category",
        "slope_all_months",
        "r2_all_months",
        "slope_excl_holiday",
        "r2_excl_holiday",
    ]
    return (
        pd.DataFrame(results, columns=columns)
        .sort_values("slope_all_months", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_trend_scatter(trend_df_extended, save_path="trend_scatter.png"):
    """Scatter of order-count trend vs. sales trend. Saves the figure, returns its path."""
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        trend_df_extended["order_count_trend"],
        trend_df_extended["sales_trend"],
        c=trend_df_extended["avg_order_value_trend"],
    )
    for _, row in trend_df_extended.iterrows():
        ax.annotate(
            row["Product Category"],
            (row["order_count_trend"], row["sales_trend"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )
    ax.set_xlabel("Order Count Trend (orders per month)")
    ax.set_ylabel("Sales Trend ($ per month)")
    ax.set_title("Revenue Growth vs. Order Volume Growth by Category")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Avg Order Value Trend ($/month)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_category_trend_grid(df, save_path="all_categories_trend.png"):
    """Actual vs. fitted monthly sales, one subplot per category. Returns the path."""
    monthly = _monthly_by_category(df)
    categories = monthly["Product Category"].unique()
    n = len(categories)
    ncols = 2
    nrows = -(-n // ncols)  # ceiling division

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), sharex=True
    )
    axes = axes.flatten() if n > 1 else [axes]

    for ax, category in zip(axes, categories):
        group = monthly[monthly["Product Category"] == category]
        _, _, fitted = _fit_line(group["Month"], group["total_sales"])

        ax.plot(group["Month"], group["total_sales"], marker="o", label="Actual")
        ax.plot(group["Month"], fitted, linestyle="--", label="Fitted")
        ax.set_title(category, fontsize=11)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Actual vs. Fitted Monthly Sales Trend by Product Category", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_full_analysis(csv_path=None):
    """Full pipeline: load -> quality report -> clean -> summarize -> trend models.

    Returns a dict of the key outputs.
    """
    raw = load_data(csv_path)
    quality = data_quality_report(raw)
    df = clean_data(raw)
    return {
        "quality": quality,
        "df": df,
        "category_summary": category_summary(df),
        "trends": monthly_category_trends(df),
        "trends_extended": monthly_category_trends_extended(df),
        "seasonality": seasonality_check(df),
    }


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)

    results = run_full_analysis()
    print("Data quality (raw data, before cleaning):")
    for key, value in results["quality"].items():
        print(f"  {key}: {value}")
    print(f"  rows after cleaning: {len(results['df'])}")
    print("\nSummary by Product Category:")
    print(results["category_summary"])
    print("\nProduct categories ranked by sales trend:")
    print(results["trends"])
    print("\nExtended trend breakdown:")
    print(results["trends_extended"])
    print("\nSeasonality check (trend with vs. without December):")
    print(results["seasonality"].round(3))

    # Plots go in their own folder so a Docker volume can be mounted there
    # (docker run -v "$PWD/output:/app/output" ...) and the PNGs land on the host.
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    plot_trend_scatter(
        results["trends_extended"],
        save_path=os.path.join(output_dir, "trend_scatter.png"),
    )
    plot_category_trend_grid(
        results["df"],
        save_path=os.path.join(output_dir, "all_categories_trend.png"),
    )
