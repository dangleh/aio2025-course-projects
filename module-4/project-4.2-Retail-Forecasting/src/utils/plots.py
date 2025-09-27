import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_sku_timeseries(df: pd.DataFrame, stock_code: str, title: str = None):
    """
    Plot weekly Quantity time series for a single SKU from the retail dataset.

    Expected columns: ['InvoiceDate', 'StockCode', 'y'] or ['Quantity'].
    """
    df_sku = df[df["StockCode"] == stock_code].sort_values("InvoiceDate")
    if df_sku.empty:
        print(f"No data for StockCode={stock_code}")
        return

    qty_col = "y" if "y" in df_sku.columns else "Quantity"

    plt.figure(figsize=(12, 4))
    plt.plot(df_sku["InvoiceDate"], df_sku[qty_col], label="Actual", color="black")
    plt.title(title or f"SKU {stock_code} – Weekly Quantity")
    plt.xlabel("Week")
    plt.ylabel("Quantity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sku_forecast(df_results: pd.DataFrame, stock_code: str,
                      y_col: str = "y_true",
                      preds: dict = None,
                      title: str = None):
    """
    Overlay actual vs predicted quantities for a single SKU.

    df_results: a DataFrame filtered to VAL/TEST that includes columns
        ['InvoiceDate','StockCode', y_col] and one or more prediction columns.
    preds: mapping {label: column_name}, e.g., {'LGBM':'y_pred_lgb', 'XGB':'y_pred_xgb'}
    """
    if preds is None:
        preds = {}

    data = df_results[df_results["StockCode"] == stock_code].sort_values("InvoiceDate")
    if data.empty:
        print(f"No rows for StockCode={stock_code}")
        return

    plt.figure(figsize=(12, 4))
    plt.plot(data["InvoiceDate"], data[y_col], label="Actual", color="black")
    for label, col in preds.items():
        if col in data.columns:
            plt.plot(data["InvoiceDate"], data[col], label=label, linestyle="--")

    plt.title(title or f"SKU {stock_code} – Actual vs Forecast")
    plt.xlabel("Week")
    plt.ylabel("Quantity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_multiple_skus(df_results: pd.DataFrame,
                       stock_codes: list,
                       y_col: str = "y_true",
                       preds: dict = None,
                       ncols: int = 3,
                       figsize=(18, 10),
                       title: str = "SKU Forecast Comparison"):
    """
    Grid plot for multiple SKUs: actual vs one or more prediction series.
    df_results should contain ['InvoiceDate','StockCode', y_col] and prediction cols.
    preds: mapping {label: column_name}
    """
    if preds is None:
        preds = {}
    n = len(stock_codes)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, sc in enumerate(stock_codes):
        ax = axes[i]
        data = df_results[df_results["StockCode"] == sc].sort_values("InvoiceDate")
        if data.empty:
            ax.axis("off")
            continue
        ax.plot(data["InvoiceDate"], data[y_col], label="Actual", color="black")
        for label, col in preds.items():
            if col in data.columns:
                ax.plot(data["InvoiceDate"], data[col], label=label, linestyle="--")
        ax.set_title(f"SKU {sc}")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

    # hide remaining axes if any
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
