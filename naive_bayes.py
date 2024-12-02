import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load training and test data
train_df = pd.read_csv('letter-trainingset.csv')
test_df = pd.read_csv('letter-predict.csv')

# Split features and labels
X_train = train_df.drop('class', axis=1)
y_train = train_df['class']
X_test = test_df.drop('class', axis=1)
y_test = test_df['class']

# Naive Bayes Classification
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Evaluation for Naive Bayes
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
class_report_nb = classification_report(y_test, y_pred_nb, output_dict=True)

# Get unique class names and sort them
class_names = sorted(y_train.unique())

# Save results to Excel
conf_matrix_df = pd.DataFrame(conf_matrix_nb, index=class_names, columns=class_names)
class_report_df = pd.DataFrame(class_report_nb).transpose()

filename = 'naive_bayes_results.xlsx'
with pd.ExcelWriter(filename) as writer:
    conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')
    class_report_df.to_excel(writer, sheet_name='Classification Report')

print("Results have been saved to", filename)