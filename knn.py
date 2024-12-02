import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import itertools

# Load training and test data
train_df = pd.read_csv('letter-trainingset.csv')
test_df = pd.read_csv('letter-predict.csv')

# Split features and labels
X_train = train_df.drop('class', axis=1)
y_train = train_df['class']
X_test = test_df.drop('class', axis=1)
y_test = test_df['class']

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parameter grids
k_values = [3, 5, 7]
weights_options = ['uniform', 'distance']
metric_options = ['euclidean', 'manhattan']
fold_numbers = [5, 10]

# Iterate over all combinations
for k, weights, metric, n_splits in itertools.product(k_values, weights_options, metric_options, fold_numbers):
    # Set up K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    knn_model = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
    cv_scores = cross_val_score(knn_model, X_train_scaled, y_train_encoded, cv=kf)
    mean_cv_score = cv_scores.mean()
    
    # Train and evaluate on test set
    knn_model.fit(X_train_scaled, y_train_encoded)
    y_pred_knn = knn_model.predict(X_test_scaled)
    conf_matrix_knn = confusion_matrix(y_test_encoded, y_pred_knn)
    class_report_knn = classification_report(y_test_encoded, y_pred_knn, output_dict=True)
    
    # Save results to Excel
    conf_matrix_df = pd.DataFrame(conf_matrix_knn, index=label_encoder.classes_, columns=label_encoder.classes_)
    class_report_df = pd.DataFrame(class_report_knn).transpose()
    cv_scores_df = pd.DataFrame({'CV Scores': cv_scores})
    mean_cv_score_df = pd.DataFrame({'Mean CV Score': [mean_cv_score]})
    
    filename = f'knn_k{k}_w{weights}_m{metric}_folds{n_splits}.xlsx'
    with pd.ExcelWriter(filename) as writer:
        conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')
        class_report_df.to_excel(writer, sheet_name='Classification Report')
        cv_scores_df.to_excel(writer, sheet_name='Cross-Validation Scores')
        mean_cv_score_df.to_excel(writer, sheet_name='Mean CV Score')

print("All combinations have been tested and results saved to Excel files.")