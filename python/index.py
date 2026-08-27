import os
import uuid
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    flash
)

from werkzeug.utils import secure_filename


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = "data-cleaning-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

CHART_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)


ALLOWED_EXTENSIONS = {
    "csv",
    "xlsx",
    "xls"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()

    return extension in ALLOWED_EXTENSIONS


def read_file(filepath):

    extension = filepath.rsplit(".", 1)[1].lower()

    if extension == "csv":

        return pd.read_csv(filepath)

    elif extension in ["xlsx", "xls"]:

        return pd.read_excel(filepath)

    else:

        raise ValueError("Unsupported file format.")


def clean_data(df):

    df = df.copy()

    original_rows = len(df)

    original_columns = len(df.columns)

    # --------------------------------------------------------
    # Remove completely empty rows
    # --------------------------------------------------------

    df = df.dropna(how="all")


    # --------------------------------------------------------
    # Remove completely empty columns
    # --------------------------------------------------------

    df = df.dropna(axis=1, how="all")


    # --------------------------------------------------------
    # Clean column names
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )


    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    duplicates_removed = df.duplicated().sum()

    df = df.drop_duplicates()


    # --------------------------------------------------------
    # Strip whitespace from text columns
    # --------------------------------------------------------

    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )


    # --------------------------------------------------------
    # Replace common missing-value strings
    # --------------------------------------------------------

    missing_values = [
        "",
        " ",
        "na",
        "n/a",
        "null",
        "none",
        "-",
        "--",
        "missing"
    ]

    df = df.replace(
        missing_values,
        np.nan
    )


    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    missing_before = int(
        df.isnull().sum().sum()
    )

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            median_value = df[column].median()

            if pd.isna(median_value):
                median_value = 0

            df[column] = df[column].fillna(
                median_value
            )

        else:

            mode_values = df[column].mode()

            if not mode_values.empty:

                replacement = mode_values.iloc[0]

            else:

                replacement = "Unknown"

            df[column] = df[column].fillna(
                replacement
            )


    # --------------------------------------------------------
    # Try converting date columns
    # --------------------------------------------------------

    for column in df.columns:

        if (
            "date" in column.lower()
            or "dob" in column.lower()
        ):

            try:

                converted = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

                valid_values = converted.notna().sum()

                if valid_values > 0:

                    df[column] = converted

            except Exception:

                pass


    # --------------------------------------------------------
    # Standardize text values
    # --------------------------------------------------------

    text_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )


    # --------------------------------------------------------
    # Cap extreme numerical outliers
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    outliers_fixed = 0

    for column in numeric_columns:

        if df[column].nunique() < 4:
            continue

        q1 = df[column].quantile(0.25)

        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        if iqr == 0:
            continue

        lower_limit = q1 - 1.5 * iqr

        upper_limit = q3 + 1.5 * iqr

        outlier_count = (
            (df[column] < lower_limit)
            |
            (df[column] > upper_limit)
        ).sum()

        outliers_fixed += int(
            outlier_count
        )

        df[column] = df[column].clip(
            lower=lower_limit,
            upper=upper_limit
        )


    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    missing_after = int(
        df.isnull().sum().sum()
    )

    cleaned_rows = len(df)

    return {
        "data": df,
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "original_columns": original_columns,
        "cleaned_columns": len(df.columns),
        "duplicates_removed": int(
            duplicates_removed
        ),
        "missing_before": missing_before,
        "missing_after": missing_after,
        "outliers_fixed": outliers_fixed
    }


# ============================================================
# CREATE CHARTS
# ============================================================

def create_charts(
    df,
    missing_data,
    job_id
):

    job_chart_folder = os.path.join(
        CHART_FOLDER,
        job_id
    )

    os.makedirs(
        job_chart_folder,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Chart 1 - Missing values
    # --------------------------------------------------------

    missing_file = os.path.join(
        job_chart_folder,
        "missing_values.png"
    )

    missing_positive = missing_data[
        missing_data > 0
    ]

    if not missing_positive.empty:

        plt.figure(
            figsize=(10, 5)
        )

        missing_positive.sort_values(
            ascending=False
        ).plot(
            kind="bar",
            color="#ef4444"
        )

        plt.title(
            "Missing Values Before Cleaning"
        )

        plt.xlabel("Columns")

        plt.ylabel("Missing Values")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            missing_file,
            dpi=120
        )

        plt.close()


    # --------------------------------------------------------
    # Chart 2 - Numerical data
    # --------------------------------------------------------

    numerical_file = os.path.join(
        job_chart_folder,
        "numerical_data.png"
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns


    if len(numeric_columns) > 0:

        columns_to_plot = list(
            numeric_columns[:4]
        )

        fig, axes = plt.subplots(
            len(columns_to_plot),
            1,
            figsize=(10, 4 * len(columns_to_plot))
        )

        if len(columns_to_plot) == 1:

            axes = [axes]


        for ax, column in zip(
            axes,
            columns_to_plot
        ):

            sns.histplot(
                df[column],
                kde=True,
                ax=ax,
                color="#2563eb"
            )

            ax.set_title(
                f"Distribution of {column}"
            )


        plt.tight_layout()

        plt.savefig(
            numerical_file,
            dpi=120
        )

        plt.close()


    # --------------------------------------------------------
    # Chart 3 - First categorical column
    # --------------------------------------------------------

    category_file = os.path.join(
        job_chart_folder,
        "category_summary.png"
    )

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns


    if len(categorical_columns) > 0:

        column = categorical_columns[0]

        counts = (
            df[column]
            .value_counts()
            .head(10)
        )

        if not counts.empty:

            plt.figure(
                figsize=(10, 5)
            )

            sns.barplot(
                x=counts.values,
                y=counts.index,
                color="#10b981"
            )

            plt.title(
                f"Top Values - {column}"
            )

            plt.xlabel("Count")

            plt.ylabel(column)

            plt.tight_layout()

            plt.savefig(
                category_file,
                dpi=120
            )

            plt.close()


    return job_chart_folder


# ============================================================
# CREATE EXCEL REPORT
# ============================================================

def create_excel_report(
    df,
    cleaning_info,
    missing_data,
    job_id
):

    filename = (
        f"cleaned_report_{job_id}.xlsx"
    )

    filepath = os.path.join(
        OUTPUT_FOLDER,
        filename
    )


    summary = pd.DataFrame({

        "Metric": [
            "Original Rows",
            "Cleaned Rows",
            "Original Columns",
            "Cleaned Columns",
            "Duplicates Removed",
            "Missing Values Before",
            "Missing Values After",
            "Outliers Fixed"
        ],

        "Value": [
            cleaning_info["original_rows"],
            cleaning_info["cleaned_rows"],
            cleaning_info["original_columns"],
            cleaning_info["cleaned_columns"],
            cleaning_info["duplicates_removed"],
            cleaning_info["missing_before"],
            cleaning_info["missing_after"],
            cleaning_info["outliers_fixed"]
        ]
    })


    statistics = df.describe(
        include="all"
    ).transpose()


    with pd.ExcelWriter(
        filepath,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="Cleaned Data",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="Report",
            index=False
        )

        statistics.to_excel(
            writer,
            sheet_name="Statistics"
        )

        missing_data.to_frame(
            name="Missing Values"
        ).to_excel(
            writer,
            sheet_name="Data Quality"
        )


    return filepath


# ============================================================
# CREATE HTML REPORT
# ============================================================

def create_html_report(
    df,
    cleaning_info,
    chart_folder,
    job_id
):

    filename = (
        f"summary_report_{job_id}.html"
    )

    filepath = os.path.join(
        OUTPUT_FOLDER,
        filename
    )


    chart_html = ""


    charts = [
        "missing_values.png",
        "numerical_data.png",
        "category_summary.png"
    ]


    for chart in charts:

        chart_path = os.path.join(
            chart_folder,
            chart
        )

        if os.path.exists(chart_path):

            relative_path = os.path.relpath(
                chart_path,
                OUTPUT_FOLDER
            )

            chart_html += f"""
            <div class="chart">
                <img src="../{relative_path}">
            </div>
            """


    preview = df.head(20).to_html(
        index=False,
        classes="data-table"
    )


    html = f"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>Data Cleaning Report</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 30px;

    font-family:
        Arial,
        sans-serif;

    background: #f1f5f9;

    color: #1e293b;
}}

.container {{
    max-width: 1200px;

    margin: auto;

    background: white;

    padding: 30px;

    border-radius: 15px;

    box-shadow:
        0 10px 30px
        rgba(0, 0, 0, 0.08);
}}

h1 {{
    color: #0f172a;
}}

h2 {{
    color: #2563eb;

    border-bottom:
        2px solid #e2e8f0;

    padding-bottom: 10px;

    margin-top: 40px;
}}

.cards {{
    display: grid;

    grid-template-columns:
        repeat(auto-fit, minmax(180px, 1fr));

    gap: 20px;

    margin: 25px 0;
}}

.card {{
    background: #f8fafc;

    padding: 20px;

    border-radius: 10px;

    text-align: center;

    border-left:
        5px solid #2563eb;
}}

.card h3 {{
    margin: 0;

    color: #64748b;

    font-size: 14px;
}}

.number {{
    font-size: 30px;

    font-weight: bold;

    color: #2563eb;

    margin-top: 10px;
}}

.chart {{
    margin: 30px 0;

    text-align: center;
}}

.chart img {{
    max-width: 100%;

    border:
        1px solid #e2e8f0;

    border-radius: 10px;
}}

.data-table {{
    width: 100%;

    border-collapse: collapse;

    font-size: 14px;
}}

.data-table th,
.data-table td {{
    border:
        1px solid #e2e8f0;

    padding: 10px;

    text-align: left;
}}

.data-table th {{
    background: #2563eb;

    color: white;
}}

.data-table tr:nth-child(even) {{
    background: #f8fafc;
}}

</style>

</head>

<body>

<div class="container">

<h1>
    Data Cleaning & Reporting Automation
</h1>

<p>
    Automatically generated data quality,
    cleaning and visualization report.
</p>


<h2>Dataset Overview</h2>

<div class="cards">

<div class="card">

<h3>Original Rows</h3>

<div class="number">
{cleaning_info["original_rows"]}
</div>

</div>


<div class="card">

<h3>Cleaned Rows</h3>

<div class="number">
{cleaning_info["cleaned_rows"]}
</div>

</div>


<div class="card">

<h3>Duplicates Removed</h3>

<div class="number">
{cleaning_info["duplicates_removed"]}
</div>

</div>


<div class="card">

<h3>Missing Values Fixed</h3>

<div class="number">
{cleaning_info["missing_before"]}
</div>

</div>


<div class="card">

<h3>Outliers Fixed</h3>

<div class="number">
{cleaning_info["outliers_fixed"]}
</div>

</div>

</div>


<h2>Cleaning Operations</h2>

<ul>

<li>
Removed duplicate records
</li>

<li>
Handled missing values
</li>

<li>
Standardized column names
</li>

<li>
Removed unnecessary whitespace
</li>

<li>
Standardized missing-value labels
</li>

<li>
Converted date columns
</li>

<li>
Handled numerical outliers
</li>

<li>
Generated statistical summaries
</li>

</ul>


<h2>Visual Analysis</h2>

{chart_html}


<h2>Cleaned Data Preview</h2>

{preview}

</div>

</body>

</html>
"""


    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(html)


    return filepath


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# UPLOAD AND PROCESS
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    if "file" not in request.files:

        flash(
            "Please select a file."
        )

        return redirect(
            url_for("index")
        )


    file = request.files["file"]


    if file.filename == "":

        flash(
            "No file selected."
        )

        return redirect(
            url_for("index")
        )


    if not allowed_file(
        file.filename
    ):

        flash(
            "Only CSV, XLSX and XLS files are allowed."
        )

        return redirect(
            url_for("index")
        )


    # --------------------------------------------------------
    # Generate unique job ID
    # --------------------------------------------------------

    job_id = uuid.uuid4().hex[:10]


    filename = secure_filename(
        file.filename
    )


    filepath = os.path.join(
        UPLOAD_FOLDER,
        f"{job_id}_{filename}"
    )


    file.save(filepath)


    try:

        # Read input
        df = read_file(filepath)


        # Store missing values BEFORE cleaning
        missing_data = df.isnull().sum()


        # Clean data
        result = clean_data(df)


        cleaned_df = result["data"]


        # Create charts
        chart_folder = create_charts(
            cleaned_df,
            missing_data,
            job_id
        )


        # Save cleaned CSV
        cleaned_csv = os.path.join(
            OUTPUT_FOLDER,
            f"cleaned_data_{job_id}.csv"
        )


        cleaned_df.to_csv(
            cleaned_csv,
            index=False
        )


        # Create Excel report
        excel_report = create_excel_report(
            cleaned_df,
            result,
            missing_data,
            job_id
        )


        # Create HTML report
        html_report = create_html_report(
            cleaned_df,
            result,
            chart_folder,
            job_id
        )


        # Render results page
        return render_template(
            "index.html",

            processed=True,

            original_rows=result[
                "original_rows"
            ],

            cleaned_rows=result[
                "cleaned_rows"
            ],

            duplicates=result[
                "duplicates_removed"
            ],

            missing_before=result[
                "missing_before"
            ],

            missing_after=result[
                "missing_after"
            ],

            outliers=result[
                "outliers_fixed"
            ],

            columns=result[
                "cleaned_columns"
            ],

            csv_file=os.path.basename(
                cleaned_csv
            ),

            excel_file=os.path.basename(
                excel_report
            ),

            html_file=os.path.basename(
                html_report
            ),

            job_id=job_id
        )


    except Exception as error:

        flash(
            f"Error processing file: {error}"
        )

        return redirect(
            url_for("index")
        )


# ============================================================
# DOWNLOAD FILE
# ============================================================

@app.route(
    "/download/<filename>"
)
def download(filename):

    safe_filename = secure_filename(
        filename
    )


    filepath = os.path.join(
        OUTPUT_FOLDER,
        safe_filename
    )


    if not os.path.exists(filepath):

        flash(
            "File not found."
        )

        return redirect(
            url_for("index")
        )


    return send_file(
        filepath,
        as_attachment=True
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print("")
    print(
        "=========================================="
    )

    print(
        " DATA CLEANING & REPORTING AUTOMATION"
    )

    print(
        "=========================================="
    )

    print(
        "Open your browser at:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("")


    app.run(
        debug=True
    )