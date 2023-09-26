import pandas as pd
from surprise.surprise import Dataset
from surprise.surprise import Reader
from surprise.surprise import SVD


# Loadind ratings file
ratings = pd.read_csv('data/ratings.csv')[['userId', 'movieId', 'rating']]

#Building the trainset
reader = Reader(rating_scale=(1,5))
dataset = Dataset.load_from_df(ratings, reader)
trainset = dataset.build_full_trainset()

#Training the model
svd = SVD()
svd.fit(trainset)

userint = int(input('Enter user id: '))
movieid = int(input('Enter movie Id: '))

svd.predict(userint, movieid)