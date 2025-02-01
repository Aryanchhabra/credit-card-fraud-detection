import matplotlib.pyplot as plt
import seaborn as sns

def plot_roc_curve(fpr, tpr, auc_score):
    """Plot ROC curve."""
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

def plot_class_distribution(df):
    """Plot class distribution of fraudulent and non-fraudulent transactions."""
    plt.figure(figsize=(6,4))
    sns.countplot(x='Class', data=df, palette='Set2')
    plt.title('Class Distribution (Fraud vs Non-Fraud)')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()
