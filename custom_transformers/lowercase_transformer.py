from sklearn.base import BaseEstimator, TransformerMixin

class LowercaseTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to convert specified columns of a DataFrame to lowercase.
    """
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        # Nothing to do here as there's no fitting process for lowering case
        return self    
    
    def transform(self, X):
        """
        Apply the lowercase transformation to the DataFrame.
        
        Parameters:
        X (pd.DataFrame): The DataFrame to modify.

        Returns:
        pd.DataFrame: The DataFrame with lowercase columns.
        """
        X = X.copy()  # Create a copy of the input DataFrame to avoid changing the original data
        for column in self.columns:
            if column in X.columns:
                X[column] = X[column].astype(str)
                X[column] = X[column].str.lower()
            else:
                print(f"Warning: '{column}' does not exist in the DataFrame.")
        return X