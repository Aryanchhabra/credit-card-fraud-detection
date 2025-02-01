from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_model(model_type, X_train, y_train):
    """Train model based on selected type."""
    if model_type == 'Logistic Regression':
        model = LogisticRegression()
    elif model_type == 'Random Forest':
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return report, auc, conf_matrix
