import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from lifelines import WeibullFitter
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Failure Forecasting App", layout="wide")
st.title("Automobile Part Failure Forecasting")

# Upload sales, returns, and future sales CSV files
sales_file = st.file_uploader("Upload Sales Data CSV", type=["csv"])
returns_file = st.file_uploader("Upload Returns Data CSV", type=["csv"])
future_sales_file = st.file_uploader("Upload Future Sales CSV (Optional)", type=["csv"])

if sales_file and returns_file:
    # Read files
    sales_data = pd.read_csv(sales_file, parse_dates=["date_in_service"])
    returns_data = pd.read_csv(returns_file, parse_dates=["date_of_return", "date_in_service"])

    st.success("Data successfully loaded.")

    # Construct failure dataset
    failure_records = []
    for _, row in returns_data.iterrows():
        for _ in range(row["quantity_returned"]):
            failure_records.append({
                "in_service": row["date_in_service"],
                "event_date": row["date_of_return"],
                "event_observed": 1
            })
    failures_df = pd.DataFrame(failure_records)

    # Construct censored dataset
    cutoff_date = pd.to_datetime("2011-09-01")
    # cutoff_date = max(sales_data["date_in_service"].max(), returns_data["date_of_return"].max())

    censored_records = []
    for _, row in sales_data.iterrows():
        total_returned = returns_data[returns_data["date_in_service"] == row["date_in_service"]]["quantity_returned"].sum()
        num_censored = int(row["quantity_in_service"] - total_returned)
        for _ in range(num_censored):
            censored_records.append({
                "in_service": row["date_in_service"],
                "event_date": cutoff_date,
                "event_observed": 0
            })
    censored_df = pd.DataFrame(censored_records)

    # Combine all data for fitting
    lifetime_df = pd.concat([failures_df, censored_df], ignore_index=True)



        ## check for NAN
    # # Check for NaNs or Infs in the duration column
    # st.write("Checking for NaNs in Duration:")
    # st.write(lifetime_df[lifetime_df["duration"].isna()])

    # st.write("Checking for Infinite values:")
    # st.write(lifetime_df[np.isinf(lifetime_df["duration"])])



    lifetime_df["event_date"] = pd.to_datetime(lifetime_df["event_date"], errors='coerce')
    lifetime_df["in_service"] = pd.to_datetime(lifetime_df["in_service"], errors='coerce')


    lifetime_df["duration"] = (lifetime_df["event_date"] - lifetime_df["in_service"]).dt.days
    lifetime_df = lifetime_df.dropna(subset=['event_date', 'duration'])




    # Fit Weibull model

    
    wf = WeibullFitter()
    wf.fit(durations=lifetime_df["duration"], event_observed=lifetime_df["event_observed"])

    st.subheader("Estimated Weibull Parameters")
    shape_param = f"{wf.rho_:.4f}"
    scale_param = f"{wf.lambda_:.4f}"
    st.write(f"Shape (rho): {shape_param}")
    st.write(f"Scale (lambda): {scale_param}")

    # Plot survival function
    st.subheader("Survival Function with Confidence Intervals")
    fig, ax = plt.subplots(figsize=(10, 6))
    wf.plot_survival_function(ci_show=True, ax=ax)
    plt.xlabel("Time (days)")
    plt.ylabel("Survival Probability")
    plt.title("Weibull Survival Function")
    st.pyplot(fig)





    # Forecasting input
    months_ahead = st.slider("Select Forecast Horizon (in months)", 1, 36, 12)

    def forecast_failures_by_cohort(sales_df, months_ahead):
        all_forecasts = []
        for _, row in sales_df.iterrows():
            cohort_date = row["date_in_service"]
            units = row["quantity_in_service"]
            for month in range(1, months_ahead + 1):
                t1 = (month - 1) * 30
                t2 = month * 30
                p1 = wf.cumulative_density_at_times(t1).values[0]
                p2 = wf.cumulative_density_at_times(t2).values[0]
                prob = max(p2 - p1, 0)
                expected_failures = units * prob
                forecast_month = cohort_date + pd.DateOffset(months=month)
                all_forecasts.append({
                    "cohort_in_service": cohort_date,
                    "forecast_month": forecast_month,
                    "expected_failures": expected_failures
                })
        return pd.DataFrame(all_forecasts)

    forecast_df = forecast_failures_by_cohort(sales_data, months_ahead=months_ahead)

    if future_sales_file:
        future_sales = pd.read_csv(future_sales_file, parse_dates=["date_in_service"])
        future_forecast_df = forecast_failures_by_cohort(future_sales, months_ahead=months_ahead)
        forecast_df = pd.concat([forecast_df, future_forecast_df], ignore_index=True)

    st.subheader("Forecasted Failures by Cohort")
    st.dataframe(forecast_df)

        # Plot forecast by calendar month


    st.subheader("Forecasted Failures by Cohort (Total Counts)")
    st.dataframe(forecast_df)

# Plot forecast by calendar month
    st.subheader("Forecasted Failures by Calendar Month")
    monthly_forecast = forecast_df.groupby("forecast_month")["expected_failures"].sum().reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(monthly_forecast["forecast_month"], monthly_forecast["expected_failures"], marker='o')
    ax2.set_xlabel("Forecast Month")
    ax2.set_ylabel("Total Expected Failures")
    ax2.set_title("Forecasted Monthly Failures (Total Counts)")
    plt.xticks(rotation=45)
    st.pyplot(fig2)


    # Download forecast CSV
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast CSV", csv, "forecast_failures.csv", "text/csv")

    # PDF Report Option
    if st.button("Download PDF Report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Automobile Part Failure Forecasting Report", ln=True, align='C')
            pdf.ln(10)
            pdf.cell(200, 10, txt=f"Weibull Shape (rho): {shape_param}", ln=True)
            pdf.cell(200, 10, txt=f"Weibull Scale (lambda): {scale_param}", ln=True)
            pdf.ln(10)
            pdf.cell(200, 10, txt="Forecast Summary (First 10 Rows):", ln=True)
            for i, row in forecast_df.head(10).iterrows():
                line = f"Cohort: {row['cohort_in_service'].date()} | Month: {row['forecast_month'].date()} | Expected Failures: {row['expected_failures']:.2f}"
                pdf.cell(200, 10, txt=line, ln=True)

            pdf.output(tmp_pdf.name)
            with open(tmp_pdf.name, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name="failure_forecast_report.pdf",
                    mime="application/pdf"
                )

else:
    st.info("Please upload both Sales and Returns CSV files to proceed.")
