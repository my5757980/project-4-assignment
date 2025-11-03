# project_4_all_assignments_streamlit.py
# FULL PROJECT 4: All Assignments in ONE FILE with Streamlit
# Panaversity Learn Modern AI Python — Complete Submission

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import json
from abc import ABC, abstractmethod
from typing import List

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Project 4: All Assignments",
    page_icon="rocket",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Project 4: Modern AI Python")
st.caption("All Assignments 00–05, Online 01, 1–6 — **One File, Streamlit-Powered!**")
st.markdown("---")

# ==============================
# SIDEBAR: Assignment Navigation
# ==============================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/code.png", width=100)
    st.title("Assignments")
    assignment = st.selectbox(
        "Choose Assignment",
        [
            "00: Python Basics",
            "01: Data Structures",
            "02: OOP Intro",
            "03: Functions & Lambdas",
            "04: File Handling",
            "05: Modules & Packages",
            "Online 01: Simple ML",
            "1: OOP Practice",
            "2: NumPy & Pandas",
            "3: Matplotlib",
            "4: Scikit-Learn",
            "5: Streamlit App",
            "6: Full AI Pipeline"
        ]
    )
    st.markdown("---")
    st.info("**Submit Repo:** [GitHub Link Here]")

# ==============================
# ASSIGNMENT TABS (All in One File)
# ==============================

# --------------------- 00 ---------------------
if assignment == "00: Python Basics":
    st.header("Assignment 00: Python Basics")
    code = '''
def calculator():
    a = float(input("First number: "))
    b = float(input("Second number: "))
    op = input("Operation (+,-,*,/): ")
    if op == '+': return a + b
    elif op == '-': return a - b
    elif op == '*': return a * b
    elif op == '/': return a / b if b != 0 else "Error"
    else: return "Invalid"
'''
    st.code(code, language='python')
    a = st.number_input("Number 1", value=10.0, key="00a")
    b = st.number_input("Number 2", value=5.0, key="00b")
    op = st.selectbox("Operation", ['+', '-', '*', '/'], key="00op")
    if st.button("Calculate", key="00calc"):
        if op == '+': res = a + b
        elif op == '-': res = a - b
        elif op == '*': res = a * b
        elif op == '/': res = a / b if b != 0 else "Error"
        st.success(f"Result: **{res}**")

# --------------------- 01 ---------------------
elif assignment == "01: Data Structures":
    st.header("Assignment 01: Data Structures")
    st.write("Dict + List + Pandas")
    data = {'Name': ['Alice', 'Bob'], 'AI_Score': [95, 88]}
    df = pd.DataFrame(data)
    st.dataframe(df)
    if st.button("Save to CSV"):
        df.to_csv("ai_scores.csv", index=False)
        st.success("Saved: `ai_scores.csv`")

# --------------------- 02 ---------------------
elif assignment == "02: OOP Intro":
    st.header("Assignment 02: OOP Intro")
    code = '''
class AIAgent:
    def __init__(self, name):
        self.name = name
    def predict(self, x):
        return x * 2
'''
    st.code(code, language='python')
    name = st.text_input("Agent Name", "Grok")
    agent = type('AIAgent', (), {'predict': lambda self, x: x * 2})()
    x = st.slider("Input", 0, 100, 50)
    st.write(f"**{name}.predict({x})** → **{agent.predict(x)}**")

# --------------------- 03 ---------------------
elif assignment == "03: Functions & Lambdas":
    st.header("Assignment 03: Functions & Lambdas")
    scores = [85, 92, 78, 96, 88]
    st.write("Scores:", scores)
    sorted_scores = sorted(scores, key=lambda x: -x)
    st.write("Sorted (desc):", sorted_scores)
    doubled = list(map(lambda x: x * 2, scores))
    st.write("Doubled:", doubled)

# --------------------- 04 ---------------------
elif assignment == "04: File Handling":
    st.header("Assignment 04: File Handling")
    content = st.text_area("Write JSON data", '{"model": "Grok", "accuracy": 0.95}')
    if st.button("Save JSON"):
        with open("model_info.json", "w") as f:
            f.write(content)
        st.success("Saved: `model_info.json`")
    if os.path.exists("model_info.json"):
        with open("model_info.json") as f:
            st.json(json.load(f))

# --------------------- 05 ---------------------
elif assignment == "05: Modules & Packages":
    st.header("Assignment 05: Modules & Packages")
    st.write("Imagine folder: `ai_utils/` → `__init__.py`, `predict.py`")
    st.code('''# ai_utils/predict.py\ndef predict(x):\n    return x ** 2''', language='python')
    x = st.number_input("Input for x²", 5)
    st.write(f"**predict({x})** → **{x**2}**")

# --------------------- Online 01 ---------------------
elif assignment == "Online 01: Simple ML":
    st.header("Online 01: Simple ML")
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 4, 6, 8, 10])
    model = LinearRegression().fit(X, y)
    new_x = st.slider("Predict for", 1, 10, 6)
    pred = model.predict([[new_x]])[0]
    st.metric("Prediction", f"{pred:.2f}")

# --------------------- 1 ---------------------
elif assignment == "1: OOP Practice":
    st.header("Assignment 1: OOP Practice")
    class Animal(ABC):
        @abstractmethod
        def speak(self): pass
    class Dog(Animal):
        def speak(self): return "Woof!"
    dog = Dog()
    st.write(f"Dog says: **{dog.speak()}**")

# --------------------- 2 ---------------------
elif assignment == "2: NumPy & Pandas":
    st.header("Assignment 2: NumPy & Pandas")
    arr = np.random.randn(5, 3)
    df = pd.DataFrame(arr, columns=['A', 'B', 'C'])
    st.dataframe(df)
    st.line_chart(df)

# --------------------- 3 ---------------------
elif assignment == "3: Matplotlib":
    st.header("Assignment 3: Matplotlib")
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.plot(x, np.cos(x), label="cos(x)")
    ax.legend()
    st.pyplot(fig)

# --------------------- 4 ---------------------
elif assignment == "4: Scikit-Learn":
    st.header("Assignment 4: Scikit-Learn")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    model = SVC().fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    st.metric("SVM Accuracy", f"{acc:.2f}")

# --------------------- 5 ---------------------
elif assignment == "5: Streamlit App":
    st.header("Assignment 5: Streamlit App")
    st.write("**You are already using it!**")
    st.balloons()

# --------------------- 6 ---------------------
elif assignment == "6: Full AI Pipeline":
    st.header("Assignment 6: Full AI Pipeline")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    model = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)
    with open("final_ai_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model trained & saved: `final_ai_model.pkl`")
    st.download_button("Download Model", data=open("final_ai_model.pkl", "rb"), file_name="final_ai_model.pkl")

# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <b>Project 4 Complete</b> | All 13 Assignments in 1 File<br>
    Made with <span style='color: #ff4b4b;'>love</span> by <b>Grok</b> | 
    <a href='https://x.ai'>xAI</a>
    </div>
    """,
    unsafe_allow_html=True
)