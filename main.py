import pandas as pd 
df = pd.read_csv("heart.csv")
print(df.head())


from sklearn.model_selection import train_test_split 

age = df["age"]
heart_attack = df["target"]

age_train, age_test, heart_attack_train, heart_attack_test = train_test_split(age, heart_attack, test_size = 0.25, random_state = 0)



from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.reshape(age_train.ravel(), (len(age_train), 1))
Y = np.reshape(heart_attack_train.ravel(), (len(heart_attack_train), 1))

classifier = LogisticRegression(random_state = 0) 
classifier.fit(X, Y)