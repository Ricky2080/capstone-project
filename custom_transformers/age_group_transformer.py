from sklearn.base import BaseEstimator, TransformerMixin

class AgegroupTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer to create a new column age_group from the defendant age.
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
        pd.DataFrame: The DataFrame with age_category.
        """
        X = X.copy()  # Create a copy of the input DataFrame to avoid changing the original data
        
        #function to transform defendant age into age_group
        def categorize_age(age):
            if age < 13:
                return '0-12'
            elif age < 18:
                return '13-17'
            elif age < 21:
                return '18-20'
            elif age < 25:
                return '21-24'
            elif age < 35:
                return '25-34'
            elif age < 45:
                return '35-44'
            elif age < 55:
                return '45-54'
            elif age < 65:
                return '55-64'
            else:
                return '65+'

        # Create the new feature 'age_group'
        X['age_group'] = X['age'].apply(categorize_age)
        return X