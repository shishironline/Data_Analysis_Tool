import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Reusable Functions

# Read file
def read_file(file_path, file_type):
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
    df_cleaned = df.drop_duplicates()
    df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
    df_cleaned = df_cleaned.dropna()
    return df_cleaned

# Describe data
def describe_data(df):
    st.write("Data Information:")
    buffer = pd.io.common.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("Statistical Summary:")
    st.write(df.describe())

# Rename columns
def rename_columns(df, new_columns):
    try:
        df = df.rename(columns=new_columns)
        return df
    except Exception as e:
        st.error(f"Error renaming columns: {e}")
        return df 

# Change data types
def change_data_types(df, column_types): 
    try:
        df = df.astype(column_types)
        return df
    except Exception as e:
        st.error(f"Error changing data types: {e}")
        return df

# Handle missing values
def handle_missing_values(df, strategy='mean'):
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'drop':
        return df.dropna()
    else:
        st.error("Invalid strategy. Choose from 'mean', 'median', 'mode', or 'drop'.")
        return df

# Handle outliers
def handle_outliers(df, column, method='iqr'):
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]
    elif method == 'zscore':
        df = df[(np.abs(stats.zscore(df[column])) < 3)]
    else:
        st.error("Invalid method. Choose 'iqr' or 'zscore'.")
    return df

# Subset DataFrame
def sub_setting(df, condition):
    return df.query(condition)

# Sample data
def sample_data(df, n, method='random'):
    if method == 'random':
        return df.sample(n=n, random_state=42)
    elif method == 'stratified':
        return df.groupby('strata').apply(lambda x: x.sample(n=n, random_state=42)).reset_index(drop=True)
    else:
        st.error("Invalid method. Choose 'random' or 'stratified'.")
        return df

# Create new column
def create_new_column(df, new_column_name, calculation):
    df[new_column_name] = calculation
    return df

# Binning
def bin_data(df, column, bins, labels):
    try:
        df[f'{column}_binned'] = pd.cut(df[column], bins=bins, labels=labels)
        return df
    except Exception as e:
        st.error(f"Error binning data: {e}")
        return df

# Replace values
def replace_values(df, column_name, to_replace, value):
    df[column_name] = df[column_name].replace(to_replace, value)
    return df

# Visualization Functions
def plot_histogram(df, column):
    plt.figure()
    sns.histplot(df[column], kde=True)
    st.pyplot(plt)

def plot_boxplot(df, column):
    plt.figure()
    sns.boxplot(df[column])
    st.pyplot(plt)

def plot_scatter_plot(df, x_column, y_column):
    plt.figure()
    sns.scatterplot(x=df[x_column], y=df[y_column])
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

        # Step 3: Clean Data
        if st.button("Clean Data"):
            df = clean_data(df)
            st.write("Cleaned Data:")
            st.dataframe(df.head())

        # Step 4: Describe Data
        if st.button("Describe Data"):
            describe_data(df)

        # Step 5: Rename Columns
        new_column_names = st.text_input("Enter new column names as a dictionary (e.g., {'old_name': 'new_name'})")
        if st.button("Rename Columns"):
            try:
                new_column_dict = eval(new_column_names)  # Converting string input to dictionary
                df = rename_columns(df, new_column_dict)
                st.write("Columns renamed successfully.")
            except Exception as e:
                st.error(f"Error renaming columns: {e}")

        # Step 6: Change Data Types
        column_types_input = st.text_input("Enter column types as a dictionary (e.g., {'column_name': 'type'})")
        if st.button("Change Data Types"):
            try:
                column_types_dict = eval(column_types_input)
                df = change_data_types(df, column_types_dict)
                st.write("Data types changed successfully.")
            except Exception as e:
                st.error(f"Error changing data types: {e}")

        # Step 7: Handle Missing Values
        missing_value_strategy = st.selectbox("Select strategy for handling missing values", ['mean', 'median', 'mode', 'drop'])
        if st.button("Handle Missing Values"):
            df = handle_missing_values(df, strategy=missing_value_strategy)
            st.write("Missing values handled.")

        # Step 8: Handle Outliers
        column_for_outliers = st.selectbox("Select a column to handle outliers", df.columns)
        outlier_method = st.selectbox("Select method for handling outliers", ['iqr', 'zscore'])
        if st.button("Handle Outliers"):
            df = handle_outliers(df, column_for_outliers, method=outlier_method)
            st.write("Outliers handled.")

        # Step 9: Subset Data
        condition = st.text_input("Enter condition for subsetting (e.g., 'column_name > value')")
        if st.button("Subset Data"):
            df_subset = sub_setting(df, condition)
            st.write("Subsetted Data:")
            st.dataframe(df_subset)

        # Step 10: Sample Data
        sample_size = st.number_input("Enter the sample size", min_value=1)
        sample_method = st.selectbox("Select sampling method", ['random', 'stratified'])
        if st.button("Sample Data"):
            df_sampled = sample_data(df, sample_size, method=sample_method)
            st.write("Sampled Data:")
            st.dataframe(df_sampled)

        # Step 11: Create New Column
        new_column_name = st.text_input("Enter new column name")
        calculation = st.text_input("Enter calculation (e.g., df['column1'] + df['column2'])")
        if st.button("Create New Column"):
            df = create_new_column(df, new_column_name, eval(calculation))
            st.write("New column created successfully.")

        # Step 12: Transform Data
        column_to_bin = st.selectbox("Select a column to bin", df.columns)
        bins_input = st.text_input("Enter bin edges as a list (e.g., [0, 10, 20])")
        labels_input = st.text_input("Enter labels for the bins as a list (e.g., ['Low', 'Medium', 'High'])")
        if st.button("Bin Data"):
            try:
                bins = eval(bins_input)
                labels = eval(labels_input)
                df = bin_data(df, column_to_bin, bins, labels)
                st.write("Data binned successfully.")
            except Exception as e:
                st.error(f"Error binning data: {e}")

        # Step 13: Replace Values
        column_to_replace = st.selectbox("Select a column to replace values", df.columns)
        to_replace_value = st.text_input("Enter value to replace")
        new_value = st.text_input("Enter new value")
        if st.button("Replace Values"):
            df = replace_values(df, column_to_replace, to_replace_value, new_value)
            st.write("Values replaced successfully.")

        # Step 14: Visualization
        st.subheader("Visualizations")
        plot_column = st.selectbox("Select column for histogram", df.columns)
        if st.button("Plot Histogram"):
            plot_histogram(df, plot_column)
        boxplot_column = st.selectbox("Select column for boxplot", df.columns)
        if st.button("Plot Boxplot"):
            plot_boxplot(df, boxplot_column)
        x_column = st.selectbox("Select X column for scatter plot", df.columns)
        y_column = st.selectbox("Select Y column for scatter plot", df.columns)
        if st.button("Plot Scatter Plot"):
            plot_scatter_plot(df, x_column, y_column)

        # Final Data Display
        st.write("Final Data:")
        st.dataframe(df)

