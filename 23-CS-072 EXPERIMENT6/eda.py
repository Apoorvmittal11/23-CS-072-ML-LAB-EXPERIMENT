import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_iris(X, y, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = y
    sns.pairplot(df, hue='species', diag_kind='hist', palette='Set1')
    plt.suptitle("Iris Dataset - Feature Relationships", y=1.02)
    plt.show()
