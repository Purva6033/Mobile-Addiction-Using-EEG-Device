# Step 2 - EDA on EEG dataset (fixed for 'label' column)
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("balanced_dataset_EEG.csv")

# 2. Show basic info
print("\n✅ Dataset Loaded!")
print("Shape of dataset (rows, columns):", df.shape)
print("\nFirst 5 rows of data:\n", df.head())

# 3. Check column names
print("\nColumns in dataset:", df.columns.tolist())

# 4. Unique classes in 'label'
if 'label' in df.columns:
    print("\nUnique classes in label column:", df['label'].unique())
else:
    print("\n⚠️ No 'label' column found! Please check column names.")

# 5. Count samples per class
if 'label' in df.columns:
    print("\nClass distribution:\n", df['label'].value_counts())

    # Bar chart of class distribution
    df['label'].value_counts().plot(kind='bar', color=['green', 'orange', 'red'])
    plt.title("Class Distribution")
    plt.xlabel("Class (0=Normal, 1=Mild, 2=Severe)")
    plt.ylabel("Number of Samples")
    plt.show()

# 6. Plot first 5 EEG signals
feature_cols = [col for col in df.columns if col != 'label']  # all except label
if len(feature_cols) > 0:
    plt.figure(figsize=(12, 6))
    for i in range(5):  # first 5 samples
        plt.plot(df[feature_cols].iloc[i].values, label=f"Sample {i}")
    plt.title("EEG Signals (First 5 Samples)")
    plt.xlabel("Feature Index")
    plt.ylabel("EEG Value")
    plt.legend()
    plt.show()
