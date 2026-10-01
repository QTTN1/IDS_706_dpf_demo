"""
Sales data analysis functions for the IDS 706 Mini Assignment using a kaggle sales dataset.

Refactored out of Week2_Mini_Assignment.ipynb into testablefunctions
"""

import matplotlib
matplotlib.use("Agg") 

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_data(csv_path=None):
    """Load the sales dataset into a DataFrame.

    If csv_path is None, downloads the dataset via kagglehub (requires
    network + Kaggle credentials)
    """
    if csv_path is None:
        import kagglehub

        path = kagglehub.dataset_download("naofilahmad/sales-datset-product-sample")
        csv_path = path + "/Sales Data.csv"
    return pd.read_csv(csv_path)


def clean_data(df):
    """Clean the raw sales DataFrame.

    - Drops the leftover index column from the CSV
    - Changes "Order Date" into a datetime
    - Strips whitespace from "City" names
    """
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["City"] = df["City"].str.strip()
    return df


def filter_high_value_orders(df, threshold=500):
    """Return only orders with Sales strictly greater than threshold."""
    return df[df["Sales"] > threshold]


def filter_by_city(df, city):
    """Return only orders from the given city."""
    return df[df["City"].str.strip() == city.strip()]


def category_summary(df):
    """Returns average price, total quantity, total sales, and order count per category."""
    return (
        df.groupby("Product Category")
        .agg(
            avg_price=("Price Each", "mean"),
            t_quantity=("Quantity Ordered", "sum"),
            t_sales=("Sales", "sum"),
            order_count=("Order ID", "count"),
        )
        .sort_values("t_sales", ascending=False)
    )


def monthly_category_trends(df):
    """Fit a linear regression model (Sales ~ Month) per category.

    Returns a DataFrame with one row per category and its monthly trend slope($/month), sorted fastest-growing category to slowest-growing category.
    """
    monthly = df.groupby(["Product Category", "Month"])["Sales"].sum().reset_index()

    results = []
    for category in monthly["Product Category"].unique():
        subset = monthly[monthly["Product Category"] == category]
        X = subset[["Month"]]
        y = subset["Sales"]
        slope = LinearRegression().fit(X, y).coef_[0]
        results.append({"Product Category": category, "monthly_trend_slope": slope})

    # columns= keeps the expected schema even when results is empty (no data)
    return (
        pd.DataFrame(results, columns=["Product Category", "monthly_trend_slope"])
        .sort_values("monthly_trend_slope", ascending=False)
        .reset_index(drop=True)
    )


def monthly_category_trends_extended(df):
    """Expands on monthly_category_trends with order-count and average-order-value trends.
    """
    monthly = (
        df.groupby(["Product Category", "Month"])
        .agg(total_sales=("Sales", "sum"), order_count=("Order ID", "count"))
        .reset_index()
    )
    monthly["avg_order_value"] = monthly["total_sales"] / monthly["order_count"]

    results = []
    for category in monthly["Product Category"].unique():
        subset = monthly[monthly["Product Category"] == category]
        X = subset[["Month"]]
        sales_slope = LinearRegression().fit(X, subset["total_sales"]).coef_[0]
        order_slope = LinearRegression().fit(X, subset["order_count"]).coef_[0]
        avg_value_slope = LinearRegression().fit(X, subset["avg_order_value"]).coef_[0]
        results.append(
            {
                "Product Category": category,
                "sales_trend": sales_slope,
                "order_count_trend": order_slope,
                "avg_order_value_trend": avg_value_slope,
            }
        )

    # columns= keeps the expected schema even when results is empty (no data)
    columns = ["Product Category", "sales_trend", "order_count_trend", "avg_order_value_trend"]
    return (
        pd.DataFrame(results, columns=columns)
        .sort_values("sales_trend", ascending=False)
        .reset_index(drop=True)
    )


def plot_trend_scatter(trend_df_extended, save_path="trend_scatter.png"):
    """Scatter plot of order-count trend vs. sales trend. Saves the figure and returns it's path."""
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
    """Grid of actual-vs-fitted monthly sales, one subplot for each category.
    Saves the figureand returns it's path."""
    monthly = df.groupby(["Product Category", "Month"])["Sales"].sum().reset_index()
    categories = monthly["Product Category"].unique()
    n = len(categories)
    ncols = 2
    nrows = -(-n // ncols) 

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), sharex=True)
    axes = axes.flatten() if n > 1 else [axes]

    for ax, category in zip(axes, categories):
        subset = monthly[monthly["Product Category"] == category].sort_values("Month")
        X = subset[["Month"]]
        y = subset["Sales"]
        predicted = LinearRegression().fit(X, y).predict(X)

        ax.plot(subset["Month"], y, marker="o", label="Actual")
        ax.plot(subset["Month"], predicted, linestyle="--", label="Fitted")
        ax.set_title(category, fontsize=11)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Actual vs. Fitted Monthly Sales Trend by Product Category", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def run_full_analysis(csv_path=None):
    """Full pipeline: load + clean + summarize + models & trends.

    Returns a dict of the key outputs.
    """
    df = load_data(csv_path)
    df = clean_data(df)
    return {
        "df": df,
        "category_summary": category_summary(df),
        "trends": monthly_category_trends(df),
        "trends_extended": monthly_category_trends_extended(df),
    }


if __name__ == "__main__":
    results = run_full_analysis()
    print("Summary by Product Category:")
    print(results["category_summary"])
    print("\nProduct categories ranked by sales trend:")
    print(results["trends"])
    print("\nExtended trend breakdown:")
    print(results["trends_extended"])
    plot_trend_scatter(results["trends_extended"])
    plot_category_trend_grid(results["df"])
