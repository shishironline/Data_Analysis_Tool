import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Reusable Functions

# Read file
def read_file(file_path, file_type):
    """Reads data from an Excel or CSV file."""
    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        else:
            st.error("Unsupported file type.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Clean data
def clean_data(df):
    """Cleans the DataFrame by handling missing values and duplicates."""
    df_cleaned = df.drop_duplicates()
    df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
    df_cleaned = df_cleaned.dropna()
    return df_cleaned

# Central Tendency
def calculate_mean(df, column):
    return df[column].mean()

def calculate_median(df, column):
    return df[column].median()

def calculate_mode(df, column):
    return df[column].mode()[0]

# Dispersion
def calculate_variance(df, column):
    return df[column].var()

def calculate_std_dev(df, column):
    return df[column].std()

def calculate_range(df, column):
    return df[column].max() - df[column].min()

# Position
def calculate_percentile(df, column, q):
    return np.percentile(df[column], q)

def calculate_quartiles(df, column):
    return np.percentile(df[column], [25, 50, 75])

# Shape
def calculate_skewness(df, column):
    return stats.skew(df[column])

def calculate_kurtosis(df, column):
    return stats.kurtosis(df[column])

# Visualization Functions
def plot_histogram(df, column):
    plt.figure()
    sns.histplot(df[column], kde=True)
    st.pyplot(plt)

def plot_boxplot(df, column):
    plt.figure()
    sns.boxplot(df[column])
    st.pyplot(plt)

def plot_bar_chart(df, column):
    plt.figure()
    df[column].value_counts().plot(kind='bar')
    st.pyplot(plt)

def plot_pie_chart(df, column):
    plt.figure()
    df[column].value_counts().plot(kind='pie')
    st.pyplot(plt)

def plot_scatter_plot(df, x_column, y_column):
    plt.figure()
    sns.scatterplot(x=df[x_column], y=df[y_column])
    st.pyplot(plt)

def plot_line_chart(df, x_column, y_column):
    plt.figure()
    plt.plot(df[x_column], df[y_column])
    st.pyplot(plt)

# Streamlit App
st.title("Advanced Data Analysis App")

# File upload widget
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['csv', 'xlsx'])
file_type = st.selectbox("Select file type", options=["csv", "excel"])

if uploaded_file is not None:
    df = read_file(uploaded_file, file_type)

    if df is not None:
        st.write("Data Preview:")
        st.dataframe(df.head())

        if st.button("Clean Data"):
            df = clean_data(df)
            st.write("Cleaned Data:")
            st.dataframe(df.head())

        column = st.selectbox("Select a column for analysis", df.columns)

        if st.button("Show Central Tendency"):
            st.write(f"Mean: {calculate_mean(df, column)}")
            st.write(f"Median: {calculate_median(df, column)}")
            st.write(f"Mode: {calculate_mode(df, column)}")

        if st.button("Show Dispersion"):
            st.write(f"Variance: {calculate_variance(df, column)}")
            st.write(f"Standard Deviation: {calculate_std_dev(df, column)}")
            st.write(f"Range: {calculate_range(df, column)}")

        if st.button("Show Position"):
            st.write(f"Quartiles: {calculate_quartiles(df, column)}")
            st.write(f"90th Percentile: {calculate_percentile(df, column, 90)}")

        if st.button("Show Shape"):
            st.write(f"Skewness: {calculate_skewness(df, column)}")
            st.write(f"Kurtosis: {calculate_kurtosis(df, column)}")

        if st.button("Show Visualization"):
            plot_choice = st.selectbox("Select plot type", ["Histogram", "Box Plot", "Bar Chart", "Pie Chart", "Scatter Plot", "Line Chart"])
            
            if plot_choice == "Histogram":
                plot_histogram(df, column)
            elif plot_choice == "Box Plot":
                plot_boxplot(df, column)
            elif plot_choice == "Bar Chart":
                plot_bar_chart(df, column)
            elif plot_choice == "Pie Chart":
                plot_pie_chart(df, column)
            elif plot_choice == "Scatter Plot":
                column_x = st.selectbox("Select X column", df.columns)
                plot_scatter_plot(df, column_x, column)
            elif plot_choice == "Line Chart":
                column_x = st.selectbox("Select X column", df.columns)
                plot_line_chart(df, column_x, column)
