'''This file creates a folder with a report on randomness tests'''

from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
ALPHA = 0.04


def normalize_results(df: pd.DataFrame) -> pd.DataFrame:

    '''Normalizes result column names and fills missing service columns'''

    normalized_df = df.copy()

    if "sequence" not in normalized_df.columns and "sequence_id" in normalized_df.columns:
        normalized_df["sequence"] = normalized_df["sequence_id"]

    if "sequence" not in normalized_df.columns:
        normalized_df["sequence"] = normalized_df.groupby("test").cumcount() + 1

    if "subtest" not in normalized_df.columns:
        normalized_df["subtest"] = ""

    if "passed" not in normalized_df.columns:
        normalized_df["passed"] = normalized_df["p_value"] >= ALPHA

    if "error" not in normalized_df.columns:
        normalized_df["error"] = ""

    return normalized_df


def load_results(csv_path: str) -> pd.DataFrame:

    '''Loads CSV results produced by the NIST runner'''

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(path)
    df["p_value"] = pd.to_numeric(df["p_value"], errors = "coerce")

    return df


def add_test_label(df: pd.DataFrame) -> pd.DataFrame:

    '''Adds full test label with subtest name if subtest exists'''

    labeled_df = df.copy()
    labeled_df = labeled_df.dropna(subset = ["p_value"]).copy()

    labeled_df["test_label"] = labeled_df.apply(
        lambda row: (f"{row['test']} [{row['subtest']}]"
            if pd.notna(row.get("subtest")) and str(row.get("subtest")).strip()
            else row["test"]), axis = 1)

    return labeled_df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:

    '''Builds summary statistics for each NIST test'''

    df = df.copy()

    if "subtest" not in df.columns:
        df["subtest"] = ""

    df["test_label"] = df.apply(lambda row: ( f"{row['test']} [{row['subtest']}]"
            if pd.notna(row["subtest"]) and str(row["subtest"]).strip()
            else row["test"]), axis = 1)

    grouped = df.groupby("test_label")  
    summary = grouped["p_value"].agg(["mean", "min", "max", "count"]).reset_index()
    summary = summary.rename(columns = {"test_label": "test"})

    pass_rates = grouped["passed"].mean().reset_index()
    pass_rates = pass_rates.rename(columns = {"test_label": "test", "passed": "pass_rate"})

    summary = summary.merge(pass_rates, on = "test")
    summary["status"] = summary["pass_rate"].apply(lambda x: "OK" if x >= 0.96 else "SUSPICIOUS")

    return summary


def save_summary(summary: pd.DataFrame, output_dir: Path) -> None:

    '''Saves summary statistics to CSV'''

    output_path = output_dir / "nist_summary.csv"
    summary.to_csv(output_path, index=False)


def plot_pass_rates(summary: pd.DataFrame, output_dir: Path) -> None:

    '''Creates a bar chart of pass rates'''

    plt.figure(figsize = (12, 6))
    plt.bar(summary["test"], summary["pass_rate"])
    plt.axhline(1.0 - ALPHA, linestyle = "--", color = "red", label = "Expected pass rate")
    plt.xticks(rotation = 45, ha = "right")
    plt.ylabel("Pass Rate")
    plt.title("NIST Test Pass Rates")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir / "pass_rates.png")
    plt.close()


def plot_pvalue_histograms(df: pd.DataFrame, output_dir: Path) -> None:

    '''Creates p-value histograms for every test'''

    clean_df = add_test_label(df)

    if clean_df.empty:
        return

    for test_name in clean_df["test_label"].unique():
        subset = clean_df[clean_df["test_label"] == test_name]
        plt.figure(figsize = (8, 5))
        plt.hist(subset["p_value"], bins = 20)
        plt.xlabel("p-value")
        plt.ylabel("Frequency")
        plt.title(f"{test_name} p-value Distribution")
        plt.tight_layout()
        safe_name = test_name.replace(" ", "_").replace("/", "_")
        plt.savefig(output_dir / f"{safe_name}_hist.png")
        plt.close()


def plot_boxplot(df: pd.DataFrame, output_dir: Path) -> None:

    '''Creates a boxplot for p-values of all tests'''

    clean_df = df.dropna(subset = ["p_value"]).copy()

    if clean_df.empty:
        return

    plt.figure(figsize = (14, 6))
    clean_df.boxplot(column = "p_value", by = "test", rot = 45)
    plt.ylabel("p-value")
    plt.title("NIST p-value Boxplot")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot.png")
    plt.close()


def plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:

    '''Creates a heatmap where rows are sequences and columns are NIST tests, cell color represents p-value'''

    clean_df = df.dropna(subset = ["p_value"]).copy()

    if clean_df.empty:
        return

    clean_df["test_label"] = clean_df.apply(
        lambda row: (f"{row['test']} [{row['subtest']}]"
            if pd.notna(row.get("subtest")) and str(row.get("subtest")).strip()
            else row["test"]), axis = 1)

    pivot = clean_df.pivot_table(index = "sequence", columns = "test_label", values="p_value", aggfunc = "mean")

    plt.figure(figsize = (max(12, len(pivot.columns) * 0.6), max(6, len(pivot.index) * 0.25)))

    plt.imshow(pivot, aspect = "auto", vmin = 0, vmax = 1)
    plt.colorbar(label = "p-value")

    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation = 45, ha = "right")
    plt.yticks(range(len(pivot.index)), pivot.index)

    plt.xlabel("NIST tests")
    plt.ylabel("Sequence")
    plt.title("NIST p-value Heatmap")
    plt.tight_layout()

    plt.savefig(output_dir / "heatmap.png")
    plt.close()


def plot_interactive_pass_rates(summary: pd.DataFrame, output_dir: Path) -> None:

    '''Creates an interactive bar chart of pass rates'''

    fig = px.bar(summary, x = "test", y = "pass_rate", color = "status", 
                 hover_data = ["mean", "min", "max", "count"], title = "Interactive NIST Pass Rates")
    fig.add_hline(y = 1.0 - ALPHA, line_dash = "dash", line_color = "red", annotation_text = "Expected pass rate")
    fig.write_html(output_dir / "interactive_pass_rates.html")


def plot_interactive_pvalue_heatmap(df: pd.DataFrame, output_dir: Path) -> None:

    '''Creates an interactive heatmap of p-values by sequence and test'''

    clean_df = df.dropna(subset = ["p_value"]).copy()

    clean_df["test_label"] = clean_df.apply(
        lambda row: (f"{row['test']} [{row['subtest']}]"
            if pd.notna(row.get("subtest")) and str(row.get("subtest")).strip()
            else row["test"]), axis = 1)

    pivot = clean_df.pivot_table(index = "sequence", columns = "test_label", values = "p_value", aggfunc = "mean")
    fig = px.imshow(pivot, color_continuous_scale = "Viridis", zmin = 0, zmax = 1,
        title = "Interactive NIST p-value Heatmap", labels = {"x": "NIST test", "y": "Sequence", "color": "p-value",})
    fig.write_html(output_dir / "interactive_heatmap.html")


def plot_interactive_pvalue_distribution(df: pd.DataFrame, output_dir: Path) -> None:

    '''Creates an interactive boxplot of p-value distributions'''

    clean_df = df.dropna(subset = ["p_value"]).copy()

    if clean_df.empty:
        return
    fig = px.box(clean_df, x = "test", y = "p_value", color = "test", points = "all",
        title = "Interactive NIST p-value Distribution", hover_data = ["sequence", "subtest", "passed"])

    fig.write_html(output_dir / "interactive_pvalue_distribution.html")


def plot_qq_plot(df: pd.DataFrame, output_dir: Path) -> None:

    '''Creates QQ plots for p-values against the expected uniform distribution'''

    clean_df = df.dropna(subset = ["p_value"]).copy()

    if clean_df.empty:
        return

    for test_name in clean_df["test"].unique():
        subset = clean_df[clean_df["test"] == test_name]["p_value"].sort_values().reset_index(drop = True)

        if len(subset) < 2:
            continue

        expected = [(i + 0.5) / len(subset) for i in range(len(subset))]

        plt.figure(figsize = (6, 6))
        plt.scatter(expected, subset, s = 18)
        plt.plot([0, 1], [0, 1], linestyle = "--", color = "red", label = "Expected uniform")
        plt.xlabel("Expected uniform quantiles")
        plt.ylabel("Observed p-value quantiles")
        plt.title(f"{test_name} QQ Plot")
        plt.legend()
        plt.tight_layout()

        safe_name = test_name.replace(" ", "_").replace("/", "_")
        plt.savefig(output_dir / f"{safe_name}_qq_plot.png")
        plt.close()


def create_report_folders(output_dir: Path) -> dict[str, Path]:

    '''Creates folders for different report artifact types'''

    folders = {
        "summary": output_dir / "summary",
        "text": output_dir / "text",
        "pass_rates": output_dir / "pass_rates",
        "histograms": output_dir / "histograms",
        "boxplots": output_dir / "boxplots",
        "heatmaps": output_dir / "heatmaps",
        "qq_plots": output_dir / "qq_plots",
        "ecdf": output_dir / "ecdf"}

    for folder in folders.values():
        folder.mkdir(parents = True, exist_ok = True)

    return folders

def plot_ecdf(df: pd.DataFrame, output_dir: Path) -> None:

    '''Creates ECDF plots for p-values against the expected uniform distribution'''

    clean_df = df.dropna(subset = ["p_value"]).copy()

    if clean_df.empty:
        return

    for test_name in clean_df["test"].unique():
        subset = clean_df[clean_df["test"] == test_name]["p_value"].sort_values().reset_index(drop = True)

        if len(subset) < 2:
            continue

        ecdf = [(i + 1) / len(subset) for i in range(len(subset))]

        plt.figure(figsize = (7, 6))
        plt.step(subset, ecdf, where = "post", label = "Observed ECDF")
        plt.plot([0, 1], [0, 1], linestyle = "--", color = "red", label = "Expected uniform")
        plt.xlabel("p-value")
        plt.ylabel("Cumulative probability")
        plt.title(f"{test_name} ECDF")
        plt.legend()
        plt.tight_layout()

        safe_name = test_name.replace(" ", "_").replace("/", "_")
        plt.savefig(output_dir / f"{safe_name}_ecdf.png")
        plt.close()


def write_text_report(summary: pd.DataFrame, output_dir: Path) -> None:

    '''Creates a human-readable text report'''

    report_path = output_dir / "report.txt"

    with open(report_path, "w") as report:
        report.write("=== NIST TEST REPORT ===\n\n")

        for i, row in summary.iterrows():
            report.write(f"Test: {row['test']}\n")
            report.write(f"  Mean p-value: {row['mean']:.6f}\n")
            report.write(f"  Min p-value: {row['min']:.6f}\n")
            report.write(f"  Max p-value: {row['max']:.6f}\n")
            report.write(f"  Pass rate: {row['pass_rate']:.4f}\n")
            report.write(f"  Status: {row['status']}\n")
            report.write("\n")

        suspicious = summary[summary["status"] == "SUSPICIOUS"]
        report.write("=== FINAL RESULT ===\n\n")

        if suspicious.empty:
            report.write("All tests look statistically normal.\n")
        else:
            report.write("Some tests look suspicious:\n\n")
            for test_name in suspicious["test"]:
                report.write(f"- {test_name}\n")


def generate_report(csv_path: str, output_dir: str = "report_output") -> None:
    
    '''Full pipeline: loads CSV -> builds statistics -> saves charts and report'''

    output = Path(output_dir)
    output.mkdir(parents = True, exist_ok = True)

    folders = create_report_folders(output)

    df = load_results(csv_path)
    df = normalize_results(df)

    summary = build_summary(df)

    save_summary(summary, folders["summary"])

    plot_pass_rates(summary, folders["pass_rates"])
    plot_interactive_pass_rates(summary, folders["pass_rates"])

    plot_pvalue_histograms(df, folders["histograms"])

    plot_boxplot(df, folders["boxplots"])
    plot_interactive_pvalue_distribution(df, folders["boxplots"])

    plot_heatmap(df, folders["heatmaps"])
    plot_interactive_pvalue_heatmap(df, folders["heatmaps"])

    plot_qq_plot(df, folders["qq_plots"])

    plot_ecdf(df, folders["ecdf"])

    write_text_report(summary, folders["text"])
    



