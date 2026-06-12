import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath, sample_frac=1.0, random_state=42):
    """
    Loads the gzip compressed CSV dataset using pandas.
    Optionally samples a fraction of rows for development.
    Prints the shape, treatment rate, visit rate, and conversion rate.

    Args:
        filepath (str): Path to the gzip compressed CSV file.
        sample_frac (float): Fraction of rows to sample. Default is 1.0.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        pd.DataFrame: The loaded (and potentially sampled) dataframe.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, compression='gzip')
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)
        
    print(f"Data shape: {df.shape}")
    
    treatment_rate = df['treatment'].mean()
    visit_rate = df['visit'].mean()
    conversion_rate = df['conversion'].mean()
    
    print(f"Treatment rate: {treatment_rate:.4f}")
    print(f"Visit rate: {visit_rate:.4f}")
    print(f"Conversion rate: {conversion_rate:.4f}")
    
    return df

def split_data(df, train_size=0.70, cal_size=0.10, test_size=0.20, random_state=42):
    """
    Performs a stratified split on the treatment column to maintain the 
    treatment/control ratio across train, calibration, and test sets.

    Args:
        df (pd.DataFrame): The input dataframe.
        train_size (float): Proportion of data for training. Default is 0.70.
        cal_size (float): Proportion of data for calibration. Default is 0.10.
        test_size (float): Proportion of data for testing. Default is 0.20.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: df_train, df_cal, df_test
    """
    # Normalize the sizes in case they don't exactly sum to 1.0
    total_size = train_size + cal_size + test_size
    train_ratio = train_size / total_size
    cal_ratio = cal_size / total_size
    test_ratio = test_size / total_size
    
    # First, split into train and a temporary (cal + test) set, stratifying by treatment.
    # This ensures both the train set and the combined remaining set have the same
    # proportion of treated units as the original dataframe.
    temp_size = cal_ratio + test_ratio
    df_train, df_temp = train_test_split(
        df, 
        test_size=temp_size, 
        random_state=random_state, 
        stratify=df['treatment']
    )
    
    # Next, split the temporary set into cal and test, stratifying by treatment again.
    # The new test_size relative to the temp set is test_ratio / (cal_ratio + test_ratio).
    # This ensures the final calibration and test sets also maintain the original treatment ratio.
    relative_test_size = test_ratio / temp_size
    df_cal, df_test = train_test_split(
        df_temp, 
        test_size=relative_test_size, 
        random_state=random_state, 
        stratify=df_temp['treatment']
    )
    
    print("\n--- Split Results ---")
    print(f"Train shape: {df_train.shape}, Treatment rate: {df_train['treatment'].mean():.4f}")
    print(f"Cal shape: {df_cal.shape}, Treatment rate: {df_cal['treatment'].mean():.4f}")
    print(f"Test shape: {df_test.shape}, Treatment rate: {df_test['treatment'].mean():.4f}")
    
    return df_train, df_cal, df_test

def get_XTY(df, outcome='conversion'):
    """
    Extracts features (X), treatment (T), and outcome (Y) arrays from the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        outcome (str): The name of the outcome column. Default is 'conversion'.

    Returns:
        tuple: X (numpy array), T (numpy array), Y (numpy array)
    """
    # Features f0 to f11
    feature_cols = [f'f{i}' for i in range(12)]
    
    X = df[feature_cols].values
    T = df['treatment'].values
    Y = df[outcome].values
    
    return X, T, Y

def standardize_features(X_train, X_cal, X_test):
    """
    Standardizes features using StandardScaler. Fits only on training data.

    Args:
        X_train (np.ndarray): Training features.
        X_cal (np.ndarray): Calibration features.
        X_test (np.ndarray): Test features.

    Returns:
        tuple: X_train_scaled, X_cal_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    
    # Fit the scaler ONLY on training data to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply the fitted scaler to calibration and test data
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_cal_scaled, X_test_scaled, scaler

if __name__ == '__main__':
    import os
    # Smoke test the pipeline
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'criteo-research-uplift-v2.1.csv.gz')
    
    print("Running preprocessing pipeline smoke test...")
    
    try:
        # Load with 10% sample
        df = load_data(data_path, sample_frac=0.1)
        
        # Split data
        df_train, df_cal, df_test = split_data(df)
        
        # Get X, T, Y for each split
        X_train, T_train, Y_train = get_XTY(df_train)
        X_cal, T_cal, Y_cal = get_XTY(df_cal)
        X_test, T_test, Y_test = get_XTY(df_test)
        
        # Standardize features
        X_train_scaled, X_cal_scaled, X_test_scaled, scaler = standardize_features(X_train, X_cal, X_test)
        
        print("\n--- Final Shapes ---")
        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_cal_scaled shape: {X_cal_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
        print("Pipeline smoke test completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists or adjust the path before running the script.")
