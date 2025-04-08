import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(train_df.info())
print(test_df.info())
print(train_df.head())
print(train_df.isnull().sum())

train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace = True)
train_df.drop(columns = ['Cabin'], inplace = True)
test_df.drop(columns = ['Cabin'], inplace = True)

plt.figure(figsize = (6, 4))
sns.countplot(x = "Survived", data = train_df, palette = "coolwarm")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize = (6, 4))
sns.countplot(x = "Survived", hue = "Sex", data = train_df, palette = "coolwarm")
plt.title("Survival Count by Gender")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize = (6, 4))
sns.countplot(x = "Survived", hue = "Pclass", data = train_df, palette = "coolwarm")
plt.title("Survival Count by Class")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize = (8, 5))
sns.histplot(train_df, x = "Age", hue = "Survived", kde = True, bins = 30, palette = "coolwarm", alpha = 0.6)
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize = (6, 4))
sns.boxplot(x = "Survived", y = "Fare", data = train_df, palette = "coolwarm")
plt.title("Fare Distribution by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Fare")
plt.ylim(0, 300)
plt.show()

plt.figure(figsize = (8, 6))
sns.heatmap(train_df.corr(numeric_only = True), annot = True, cmap = "coolwarm", fmt = ".2f", linewidths = 0.5)
plt.title("Feature Correlation Heatmap")
plt.show()
