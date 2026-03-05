import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Handle missing values
df["Sleep_Disorder"] = df["Sleep_Disorder"].fillna("None")

# 🔥 Fix Blood Pressure column (important)
if "Blood_Pressure" in df.columns:
    df[["Systolic_BP", "Diastolic_BP"]] = df["Blood_Pressure"].str.split("/", expand=True)
    df["Systolic_BP"] = pd.to_numeric(df["Systolic_BP"], errors="coerce")
    df["Diastolic_BP"] = pd.to_numeric(df["Diastolic_BP"], errors="coerce")
    df = df.drop("Blood_Pressure", axis=1)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# -----------------------------
# DEFINE FEATURES & TARGET (VERY IMPORTANT)
# -----------------------------
y = df["Quality_of_Sleep"]          # Target (what we predict)
X = df.drop("Quality_of_Sleep", axis=1)  # Features (inputs)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy Check
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# 🔮 USER INPUT PREDICTION (CORRECT VERSION)
# -----------------------------
print("\n===== Sleep Quality Prediction System =====")

try:
    age = int(input("Enter Age: "))
    stress = int(input("Enter Stress Level (1-10): "))
    sleep_duration = float(input("Enter Sleep Duration (hours): "))
    physical_activity = int(input("Enter Physical Activity Level: "))
    heart_rate = int(input("Enter Heart Rate: "))

    # Create input with same structure as training data
    user_data = X.mean().to_frame().T  # base row with correct columns

    # Update important features (only if they exist in dataset)
    if "Age" in user_data.columns:
        user_data["Age"] = age
    if "Stress_Level" in user_data.columns:
        user_data["Stress_Level"] = stress
    if "Sleep_Duration" in user_data.columns:
        user_data["Sleep_Duration"] = sleep_duration
    if "Physical_Activity_Level" in user_data.columns:
        user_data["Physical_Activity_Level"] = physical_activity
    if "Heart_Rate" in user_data.columns:
        user_data["Heart_Rate"] = heart_rate

    # Predict
    prediction = model.predict(user_data)

    print("\n🧠 Predicted Sleep Quality:", int(prediction[0]))
    sleep_score = int(prediction[0])

    print("\n----- Sleep Analysis Result -----")

    if sleep_score >= 8:
        print("""
    Your sleep quality appears to be very good. This indicates that your lifestyle habits
    are supporting healthy sleep patterns. Maintaining regular sleep duration, moderate
    stress levels, and consistent physical activity can help you continue enjoying
    high-quality sleep.

    Recommendation:
    Continue your current routine, maintain consistent sleep schedules,
    and avoid excessive screen time before bedtime.
    """)

    elif sleep_score >= 6:
        print("""
    Your sleep quality is moderate. While your sleep is generally acceptable,
    there may be some lifestyle factors slightly affecting your rest.

    Possible factors could include mild stress levels, inconsistent sleep duration,
    or insufficient physical activity.

    Recommendations:
    • Try to maintain a consistent sleep schedule
    • Engage in regular physical activity
    • Reduce stress through relaxation techniques
    • Limit screen exposure before sleeping
    """)

    else:
        print("""
    Your predicted sleep quality is low. This suggests that certain lifestyle
    factors may be negatively affecting your sleep.

    Common causes of poor sleep may include high stress levels, inadequate
    sleep duration, lack of physical activity, or irregular sleeping habits.

    Recommendations for improving sleep quality:
    • Aim for 7–9 hours of sleep every night
    • Reduce stress through meditation or relaxation exercises
    • Increase daily physical activity
    • Avoid caffeine and heavy meals before bedtime
    • Reduce mobile or laptop screen time at night
    • Maintain a consistent bedtime routine
    """)

except Exception as e:
    print("Error in input:", e)