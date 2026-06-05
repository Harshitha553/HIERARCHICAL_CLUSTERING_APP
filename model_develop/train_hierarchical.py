import pandas as pd
import pickle

from sklearn.cluster import AgglomerativeClustering

from backend.preprocess import preprocess_data

df = pd.read_csv("datasets/Mall_Customers.csv")

X_scaled = preprocess_data(df)

model = AgglomerativeClustering(
    n_clusters=5,
    linkage='ward'
)

labels = model.fit_predict(X_scaled)

with open("hierarchical_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Hierarchical Model Saved Successfully")