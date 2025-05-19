from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

#Load dataset
data = Dataset.load_builtin('ml-100k')
print(data)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)
print("RMSE: ",accuracy.rmse(predictions))
print("MAE: ",accuracy.mae(predictions))

uid = str(196)
iid = str(302)
pred = model.predict(uid, iid)
print(f"Predicted rating by User {uid} for Movie {iid}: {pred.est:.2f}")

# ✅ Get top-N recommendations for all users
from collections import defaultdict

def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=5)

# 🎯 Top 5 for User 196
print(f"\nTop 5 movie recommendations for User 196:")
for movie_id, score in top_n["196"]:
    print(f"Movie ID: {movie_id}, Predicted Rating: {score:.2f}")

# ✅ Sample product descriptions
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Mock dataset of products
df = pd.DataFrame({
    'product_id': ['101', '102', '103', '104', '105'],
    'title': ['Wireless Mouse', 'Gaming Keyboard', 'Noise Cancelling Headphones', 'USB-C Hub', 'Portable SSD'],
    'description': [
        "Smooth wireless mouse with ergonomic design",
        "Mechanical keyboard with RGB lighting",
        "Headphones with active noise cancellation and deep bass",
        "Multiport USB-C hub for Mac and Windows",
        "Fast and durable SSD for file storage and transfer"
    ]
})

# ✅ TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# ✅ Cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# ✅ Recommendation function
def recommend(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]
    product_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[product_indices]

# 🎯 Recommend similar products
print("\nProducts similar to 'Gaming Keyboard':")
print(recommend('Gaming Keyboard'))
