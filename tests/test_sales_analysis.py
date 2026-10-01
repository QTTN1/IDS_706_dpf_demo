"""
Tests for sales_analysis.py(matches week2_mini_assignment.ipynb).
"""

import pandas as pd
import pytest
from sales_analysis import (
    category_summary,
    clean_data,
    filter_by_city,
    filter_high_value_orders,
    monthly_category_trends,
    monthly_category_trends_extended,
    run_full_analysis,
)


def _make_og_rows():
    """A couple of rows shaped like the real original CSV export, before cleaning:
    - has the leftover 'Unnamed: 0' index column
    - Order Date is a plain string, not a real date yet
    - City has stray whitespace (it's parsed out of a longer address string in the real data, and sometimes ends up as " Boston" instead of "Boston")
      -> to be used in tests for clean_data and filter_by_city to ensure output is cleaned properly.
    """
    return [
        {
            "Unnamed: 0": 0,
            "Order ID": 100,
            "Product Category": "Laptops",
            "Product": "Laptops Item",
            "Quantity Ordered": 1,
            "Price Each": 500,
            "Order Date": "1/15/19 10:00",
            "Purchase Address": "1 Main St, Boston, MA 00000",
            "Month": 1,
            "Sales": 500,
            "City": " Boston",
            "Hour": 10,
            "Time of Day": "Morning",
        },
        {
            "Unnamed: 0": 1,
            "Order ID": 101,
            "Product Category": "Batteries",
            "Product": "Batteries Item",
            "Quantity Ordered": 1,
            "Price Each": 20,
            "Order Date": "1/16/19 11:00",
            "Purchase Address": "2 Main St, Seattle, WA 00000",
            "Month": 1,
            "Sales": 20,
            "City": "Seattle ",
            "Hour": 11,
            "Time of Day": "Morning",
        },
    ]


@pytest.fixture
def raw_df():
    """Transforms csv data into an uncleaned dataframe. Still has Unnamed: 0, string dates, whitespace in City."""
    return pd.DataFrame(_make_og_rows())


def test_clean_data(raw_df):
    cleaned = clean_data(raw_df.copy())

    # the extra index column from the CSV export should be gone
    assert "Unnamed: 0" not in cleaned.columns

    # "Order Date" should now be a real datetime, not a string
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Order Date"])

    # "City" whitespace should be stripped: " Boston" -> "Boston"
    assert set(cleaned["City"].unique()) == {"Boston", "Seattle"}


@pytest.fixture
def clean_df(raw_df):
    """Cleaned version of raw_df."""
    return clean_data(raw_df.copy())


def test_filter_by_city(clean_df):
    boston_only = filter_by_city(clean_df, "Boston")

    # only Boston rows should remain since searched for Boston
    assert set(boston_only["City"].unique()) == {"Boston"}

    # exactly one row matches (only the Laptops row is Boston)
    assert len(boston_only) == 1


def _make_trend_test_rows():
    """Three months of data for two categories, trending in opposite directions to test monthly_category_trends:
    - Laptops: Sales rising 500, 1000, 1500 (positive slope)
    - Batteries: Sales falling 300, 200, 100 (negative slope)
    """
    return [
        {"Product Category": "Laptops", "Month": 1, "Sales": 500},
        {"Product Category": "Laptops", "Month": 2, "Sales": 1000},
        {"Product Category": "Laptops", "Month": 3, "Sales": 1500},
        {"Product Category": "Batteries", "Month": 1, "Sales": 300},
        {"Product Category": "Batteries", "Month": 2, "Sales": 200},
        {"Product Category": "Batteries", "Month": 3, "Sales": 100},
    ]


@pytest.fixture
def trend_df():
    """Transforms sample data for testing monthly category trends."""
    return pd.DataFrame(_make_trend_test_rows())


def test_monthly_category_trends(trend_df):
    trends = monthly_category_trends(trend_df)

    # both categories should show up, one row each
    assert set(trends["Product Category"]) == {"Laptops", "Batteries"}

    # Laptops slope should be +$500/month
    laptops_slope = trends.loc[
        trends["Product Category"] == "Laptops", "monthly_trend_slope"
    ].iloc[0]
    assert laptops_slope == pytest.approx(500)

    # Batteries slope should be -$100/month
    batteries_slope = trends.loc[
        trends["Product Category"] == "Batteries", "monthly_trend_slope"
    ].iloc[0]
    assert batteries_slope == pytest.approx(-100)

    # sorted fastest-growing first -> Laptops (positive) before Batteries (negative)
    assert trends.iloc[0]["Product Category"] == "Laptops"


@pytest.fixture
def sample_csv(tmp_path):
    """Writes a small CSV to disk and returns its path.
    run_full_analysis reads this instead of 'Sales Data.csv' so this test never touches
    kagglehub or the network.
    """
    rows = [
        {
            "Order ID": 100,
            "Product Category": "Laptops",
            "Product": "Laptops Item",
            "Quantity Ordered": 1,
            "Price Each": 500,
            "Order Date": "1/15/19 10:00",
            "Purchase Address": "1 Main St, Boston, MA 00000",
            "Month": 1,
            "Sales": 500,
            "City": " Boston",
            "Hour": 10,
            "Time of Day": "Morning",
        },
        {
            "Order ID": 101,
            "Product Category": "Laptops",
            "Product": "Laptops Item",
            "Quantity Ordered": 1,
            "Price Each": 600,
            "Order Date": "2/15/19 10:00",
            "Purchase Address": "1 Main St, Boston, MA 00000",
            "Month": 2,
            "Sales": 600,
            "City": " Boston",
            "Hour": 10,
            "Time of Day": "Morning",
        },
        {
            "Order ID": 200,
            "Product Category": "Batteries",
            "Product": "Batteries Item",
            "Quantity Ordered": 1,
            "Price Each": 20,
            "Order Date": "1/16/19 11:00",
            "Purchase Address": "2 Main St, Seattle, WA 00000",
            "Month": 1,
            "Sales": 20,
            "City": "Seattle ",
            "Hour": 11,
            "Time of Day": "Morning",
        },
        {
            "Order ID": 201,
            "Product Category": "Batteries",
            "Product": "Batteries Item",
            "Quantity Ordered": 1,
            "Price Each": 15,
            "Order Date": "2/16/19 11:00",
            "Purchase Address": "2 Main St, Seattle, WA 00000",
            "Month": 2,
            "Sales": 15,
            "City": "Seattle ",
            "Hour": 11,
            "Time of Day": "Morning",
        },
    ]
    csv_path = tmp_path / "sample_sales.csv"
    pd.DataFrame(rows).to_csv(csv_path)
    return csv_path


def test_run_full_analysis(sample_csv):
    """System/integration test: runs the whole pipeline (load -> clean ->
    summarize -> trend -> trend extended) end to end from a CSV file, the
    way run_full_analysis is actually used. Each assertion below checks
    that ONE stage of the pipeline handed correct data to the next."""
    results = run_full_analysis(csv_path=str(sample_csv))

    # verify load_data + clean_data
    # confirms load_data() read the CSV and clean_data() actually ran
    assert "Unnamed: 0" not in results["df"].columns
    assert set(results["df"]["City"].unique()) == {"Boston", "Seattle"}

    # verify category_summary
    # confirms the cleaned df was correctly passed into category_summary
    # & confirms both categories survived grouping
    assert set(results["category_summary"].index) == {"Laptops", "Batteries"}

    # verify monthly_category_trends
    # confirms the cleaned df was correctly passed into monthly_category_trends
    assert set(results["trends"]["Product Category"]) == {"Laptops", "Batteries"}

    # verify monthly_category_trends_extended
    # confirms the cleaned df was correctly passed into monthly_category_trends_extended
    # confirms the 2nd call didn't get skipped or incorrect data
    assert set(results["trends_extended"]["Product Category"]) == {
        "Laptops",
        "Batteries",
    }


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.fixture
def boundary_orders_df():
    """Three orders straddling the $500 threshold: just under, exactly at, just over."""
    return pd.DataFrame({"Order ID": [1, 2, 3], "Sales": [499.99, 500.00, 500.01]})


@pytest.fixture
def empty_raw_df(raw_df):
    """Same columns as raw_df but zero rows -- simulates an empty CSV export."""
    return raw_df.iloc[0:0]


def test_filter_high_value_orders_boundary(boundary_orders_df):
    """Edge case: the threshold is strict (Sales > 500), so an order of
    exactly $500 must be excluded, while $500.01 is kept."""
    high_value = filter_high_value_orders(boundary_orders_df, threshold=500)

    assert list(high_value["Order ID"]) == [3]
    assert 500.00 not in high_value["Sales"].values


def test_filter_by_city_nonexistent(clean_df):
    """Edge case: a city that isn't in the data should return an empty
    DataFrame (same columns), not raise an error."""
    result = filter_by_city(clean_df, "Atlantis")

    assert result.empty
    assert list(result.columns) == list(clean_df.columns)


def test_empty_dataframe(empty_raw_df):
    """Edge case: an empty dataset (correct columns, zero rows) should flow
    through cleaning, summarizing, and trend fitting without crashing, and
    each step should return an empty result with the expected columns."""
    cleaned = clean_data(empty_raw_df.copy())
    assert cleaned.empty
    assert "Unnamed: 0" not in cleaned.columns

    assert category_summary(cleaned).empty

    trends = monthly_category_trends(cleaned)
    assert trends.empty
    assert "monthly_trend_slope" in trends.columns

    trends_ext = monthly_category_trends_extended(cleaned)
    assert trends_ext.empty
    assert {"sales_trend", "order_count_trend", "avg_order_value_trend"} <= set(
        trends_ext.columns
    )
