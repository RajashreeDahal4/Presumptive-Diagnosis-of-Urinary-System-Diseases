#!/usr/bin/env python
# coding: utf-8

# Acute Inflammations Data using Logistic Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import seaborn as sns


# Data Preprocessing
data = pd.read_csv(
    "diagnosis.data",
    sep="\t",
    encoding="utf-16",
    header=None,
    names=[
        "Temperature",
        "Nausea",
        "Lumbar Pain",
        "Urine Pushing",
        "Micturition pains",
        "Burning",
        "Inflammation",
        "Nephritis",
    ],
)


def temperature(temperature):
    if "," in temperature:
        temperature = temperature.replace(",", ".")
    return temperature


data["Temperature"] = data.apply(lambda row: temperature(row["Temperature"]), axis=1)

data["Temperature"] = data["Temperature"].astype("float")


def boolean_conv(info):
    if info == "no":
        info = 0
    else:
        info = 1
    return info


data["Nausea"] = data.apply(lambda row: boolean_conv(row["Nausea"]), axis=1)
data["Lumbar Pain"] = data.apply(lambda row: boolean_conv(row["Lumbar Pain"]), axis=1)
data["Urine Pushing"] = data.apply(
    lambda row: boolean_conv(row["Urine Pushing"]), axis=1
)
data["Micturition pains"] = data.apply(
    lambda row: boolean_conv(row["Micturition pains"]), axis=1
)
data["Burning"] = data.apply(lambda row: boolean_conv(row["Burning"]), axis=1)
data["Inflammation"] = data.apply(lambda row: boolean_conv(row["Inflammation"]), axis=1)
data["Nephritis"] = data.apply(lambda row: boolean_conv(row["Nephritis"]), axis=1)


# Train the model
X = data.iloc[:, :-2]
y_inflammation = data["Inflammation"]
y_nephritis = data["Nephritis"]

# Split the data into train and test sets
(
    X_train,
    X_test,
    y_train_inflammation,
    y_test_inflammation,
    y_train_nephritis,
    y_test_nephritis,
) = train_test_split(X, y_inflammation, y_nephritis, test_size=0.2, random_state=42)

# Initialize and train logistic regression models
logreg_inflammation = LogisticRegression()
logreg_inflammation.fit(X_train, y_train_inflammation)

logreg_nephritis = LogisticRegression()
logreg_nephritis.fit(X_train, y_train_nephritis)

# Make predictions on test data
y_pred_inflammation = logreg_inflammation.predict(X_test)
y_pred_nephritis = logreg_nephritis.predict(X_test)

# Confusion Matrix

target_names = ["without Inflammation", "with Inflammation"]
print(
    classification_report(
        y_test_inflammation, y_pred_inflammation, target_names=target_names
    )
)

target_names = ["without Nephritis", "with Nephritis"]
print(
    classification_report(y_test_nephritis, y_pred_nephritis, target_names=target_names)
)

# Create confusion matrix
confusion_matrix_inflammation = pd.crosstab(
    y_test_inflammation,
    y_pred_inflammation,
    rownames=["Actual"],
    colnames=["Predicted"],
)
confusion_matrix_nephritis = pd.crosstab(
    y_test_nephritis, y_pred_nephritis, rownames=["Actual"], colnames=["Predicted"]
)

# Plot confusion matrix
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix_inflammation, annot=True, cmap="YlGnBu")
plt.title("Confusion Matrix (Inflammation of Urinary Bladder)")
plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix_nephritis, annot=True, cmap="YlGnBu")
plt.title("Confusion Matrix (Nephritis of Renal Pelvis)")
plt.tight_layout()
plt.show()


# ROC Curve

# Inflammation of Urinary Bladder
fpr_inflammation, tpr_inflammation, thresholds_inflammation = roc_curve(
    y_test_inflammation, logreg_inflammation.predict_proba(X_test)[:, 1]
)
roc_auc_inflammation = auc(fpr_inflammation, tpr_inflammation)

# Nephritis of Renal Pelvis
fpr_nephritis, tpr_nephritis, thresholds_nephritis = roc_curve(
    y_test_nephritis, logreg_nephritis.predict_proba(X_test)[:, 1]
)
roc_auc_nephritis = auc(fpr_nephritis, tpr_nephritis)


# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(
    fpr_inflammation,
    tpr_inflammation,
    label="Inflammation of Urinary Bladder (AUC = {:.2f})".format(roc_auc_inflammation),
)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve for Inflammation")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(
    fpr_nephritis,
    tpr_nephritis,
    label="Nephritis of Renal Pelvis (AUC = {:.2f})".format(roc_auc_nephritis),
)
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve for Nephritis")
plt.legend()
plt.show()


# Predicted

# Plot bar graph for predicted values Inflammation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(
    ["No Inflammation", "Inflammation"],
    [len(y_pred_inflammation) - sum(y_pred_inflammation), sum(y_pred_inflammation)],
    color=["blue", "red"],
)
ax1.set_title("Inflammation of Urinary Bladder (Predicted)")

# Plot bar graph for actual values Inflammation
ax2.bar(
    ["No Inflammation", "Inflammation"],
    [len(y_test_inflammation) - sum(y_test_inflammation), sum(y_test_inflammation)],
    color=["blue", "red"],
)
ax2.set_title("Inflammation of Urinary Bladder (Actual)")
plt.show()

# Plot bar graph for predicted values nephritis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.bar(
    ["No Nephritis", "Nephritis"],
    [len(y_pred_nephritis) - sum(y_pred_nephritis), sum(y_pred_nephritis)],
    color=["blue", "red"],
)
ax1.set_title("Inflammation of Urinary Bladder (Predicted)")

# Plot bar graph for actual values nephritis
ax2.bar(
    ["No Nephritis", "Nephritis"],
    [len(y_test_nephritis) - sum(y_test_nephritis), sum(y_test_nephritis)],
    color=["blue", "red"],
)
ax2.set_title("Inflammation of Urinary Bladder (Actual)")
plt.show()

inflammation_accuracy = accuracy_score(y_test_inflammation, y_pred_inflammation)
inflammation_precision = precision_score(y_test_inflammation, y_pred_inflammation)
inflammation_recall = recall_score(y_test_inflammation, y_pred_inflammation)
inflammation_f1_score = f1_score(y_test_inflammation, y_pred_inflammation)

nephritis_accuracy = accuracy_score(y_test_nephritis, y_pred_nephritis)
nephritis_precision = precision_score(y_test_nephritis, y_pred_nephritis)
nephritis_recall = recall_score(y_test_nephritis, y_pred_nephritis)
nephritis_f1_score = f1_score(y_test_nephritis, y_pred_nephritis)

# Plot the evaluation metrics using seaborn
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
sns.barplot(
    x=["Accuracy", "Precision", "Recall", "F1-score"],
    y=[
        inflammation_accuracy,
        inflammation_precision,
        inflammation_recall,
        inflammation_f1_score,
    ],
    ax=axes[0],
)
sns.barplot(
    x=["Accuracy", "Precision", "Recall", "F1-score"],
    y=[nephritis_accuracy, nephritis_precision, nephritis_recall, nephritis_f1_score],
    ax=axes[1],
)
axes[0].set_title("Inflammation of Urinary Bladder")
axes[1].set_title("Nephritis of Renal Pelvis")
axes[0].set_xlabel("Metrics")
axes[1].set_xlabel("Metrics")
axes[0].set_ylabel("Score")
axes[1].set_ylabel("Score")

plt.tight_layout()
plt.show()
