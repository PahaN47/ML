import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score

GLOBAL = {"target_key": "Purchased", "feature_names": []}


@st.cache_data
def load_data():
    data = pd.read_csv("~/MO-lab/lab-6/car_data.csv")
    le = LabelEncoder()
    mms = MinMaxScaler()
    data["Age"] = mms.fit_transform(data[["Age"]])
    data["AnnualSalary"] = mms.fit_transform(data[["AnnualSalary"]])
    data["Gender"] = le.fit_transform(data["Gender"])
    data = data.drop(columns=["User ID"])
    data.head()
    return data


@st.cache_data
def preprocess_data(_):
    X_data, y_data = (
        data.drop(columns=[GLOBAL["target_key"]]),
        data[[GLOBAL["target_key"]]],
    )
    X_test, X_train, y_test, y_train = train_test_split(X_data, y_data)

    GLOBAL["feature_names"] = [column for column in X_test.columns]
    return X_test, X_train, y_test, y_train


def class_accuracy_score(y_true: pd.DataFrame, y_pred: np.ndarray):
    unique, counts = np.unique(y_true[GLOBAL["target_key"]], return_counts=True)
    y_true_values = y_true[GLOBAL["target_key"]].values
    result = {}
    for value in unique:
        result[value] = 0
    for index in range(len(y_true_values)):
        if y_true_values[index] == y_pred[index]:
            result[y_true_values[index]] += 1
    for index in range(len(unique)):
        result[unique[index]] /= counts[index]
    result["all"] = accuracy_score(y_true, y_pred)
    return result


st.sidebar.header("Logistic Regression")
data = load_data()

solver = None
penalty = None
l1_ratio = None
max_iter = None
random_state = None
class_weight = None
regularization_strength = None

SOLVER_PENALTIES = {
    "lbfgs": ["l2", None],
    "liblinear": ["l1", "l2"],
    "newton-cg": ["l2", None],
    "newton-cholesky": ["l2", None],
    "sag": ["l2", None],
    "saga": ["elasticnet", "l1", "l2", None],
}

regularization_strength = st.sidebar.slider(
    "regularization_strength", min_value=0.1, max_value=10.0, step=0.1, value=1.0
)

solver = st.sidebar.radio(
    "solver", ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
)
penalty = st.sidebar.radio("penalty", SOLVER_PENALTIES[solver])

if penalty == "elasticnet":
    l1_ratio = st.sidebar.slider("l1_ratio", min_value=0.0001, max_value=1.0)

max_iter = st.sidebar.slider("max_iter", min_value=1, max_value=1000, value=100)

if solver == "liblinear" or solver == "sag" or solver == "saga":
    random_state = st.sidebar.slider("random_state", min_value=0, max_value=100)

if st.sidebar.checkbox("balanced_class_weights"):
    class_weight = "balanced"


my_cmap = sns.color_palette("vlag", as_cmap=True)
if st.checkbox("Show correlation matrix"):
    fig1, ax = plt.subplots()
    sns.heatmap(data.corr(), cmap=my_cmap, ax=ax, annot=True, fmt=".2f")
    st.pyplot(fig1)

data_len = data.shape[0]

X_train, X_test, y_train, y_test = preprocess_data(data)
lr = LogisticRegression(
    C=1 / regularization_strength,
    solver=solver,
    penalty=penalty,
    l1_ratio=l1_ratio,
    max_iter=max_iter,
    random_state=random_state,
    class_weight=class_weight,
)
lr.fit(X_train, y_train)

st.header("Model quality")

y_pred = lr.predict(X_test)

st.subheader("Accuracy")
acc_df = pd.DataFrame.from_dict(
    class_accuracy_score(y_test, y_pred), orient="index", columns=["accuracy"]
)
st.dataframe(acc_df)

st.subheader("Confusion matrix")
fig2, ax = plt.subplots(figsize=(10, 5))
cm = confusion_matrix(y_test, y_pred, normalize="all")
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax, cmap=my_cmap)
fig2.suptitle("Confusion Matrix")
st.pyplot(fig2)


def draw_roc_curve(y_true, y_score, ax, pos_label=1, average="micro"):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    lw = 2
    ax.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc_value,
    )
    ax.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right")


st.subheader("ROC curve")
fig3, ax = plt.subplots(figsize=(10, 5))
draw_roc_curve(y_test.values, y_pred, ax)
st.pyplot(fig3)
