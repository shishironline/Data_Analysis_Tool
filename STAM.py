import streamlit as st
import pandas as pd
from io import StringIO

# Reusable functions from your notebook

def read_file(file_path, file_type):
    """
    Reads data from an Excel or CSV file.

    Parameters:
    file_path (str): The path to the file.
    file_type (str): The type of file ('excel' or 'csv').

    Returns:
    DataFrame: A pandas DataFrame containing the file data.
    """
    try:
        if file_type.lower() == 'excel':
            df = pd.read_excel(file_path)
        elif file_type.lower() == 'csv':
            df = pd.read_csv(file_path)
        else:
            st.error("Unsupported file type. Please choose 'excel' or 'csv'.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading {file_type} file: {e}")
        return None

def describe_data(df):
    """
    Describes the data using basic information and statistical summary.

    Parameters:
    df (pd.DataFrame): The DataFrame to describe.

    Returns:
    None: Prints the info and description of the DataFrame.
    """
    st.write("Data Info:")
    st.write(df.info())
    st.write("Statistical Summary:")
    st.write(df.describe())

def rename_columns(df, new_columns):
    """
    Renames columns in a DataFrame.

    Parameters:
    df (DataFrame): The input pandas DataFrame.
    new_columns (dict): A dictionary mapping old column names to new column names.

    Returns:
    DataFrame: A pandas DataFrame with renamed columns.
    """
    try:
        df = df.rename(columns=new_columns)
        st.success("Columns renamed successfully!")
        return df
    except Exception as e:
        st.error(f"Error renaming columns: {e}")
        return df

# Streamlit app layout

st.title("Data Analysis App")

# File upload widget
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=['csv', 'xlsx'])

# Let user select the type of file
file_type = st.selectbox("Select file type", options=["csv", "excel"])

# If file is uploaded, read and show basic description
if uploaded_file is not None:
    # Convert the file to a DataFrame
    df = read_file(uploaded_file, file_type)
    
    if df is not None:
        # Show a preview of the DataFrame
        st.write("Data Preview:")
        st.dataframe(df.head())

        # Display file description
        if st.button("Describe Data"):
            describe_data(df)
        
        # Allow renaming of columns
        st.subheader("Rename Columns")
        columns = df.columns.tolist()
        st.write("Current Columns: ", columns)

        new_columns = {}
        for col in columns:
            new_col_name = st.text_input(f"Rename '{col}' to:", value=col)
            new_columns[col] = new_col_name
        
        if st.button("Rename Columns"):
            df = rename_columns(df, new_columns)
            st.write("Updated DataFrame:")
            st.dataframe(df.head())

