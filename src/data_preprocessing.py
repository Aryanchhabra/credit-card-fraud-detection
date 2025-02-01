import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(url):
    """Load the dataset."""
    return pd.read_csv(url)

def preprocess_data(df):
    """Preprocess data and scale features."""
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y

def split_data(X, y):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=0.3, random_state=42)
