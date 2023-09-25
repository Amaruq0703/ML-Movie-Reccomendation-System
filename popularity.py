import pandas as pd

movies = pd.read_csv('data/movies.csv')
credits = pd.read_csv('data/credits.csv')
ratings = pd.read_csv('data/ratings.csv')

# Popularity Based filtering

"""
imdb_weight_formula = (v/(v+m))*R + (m /(v+m))*C
v- no. of votes
m- minimum no. of votes for a movie to be popular
R - average movie rating
C - average rating across all movies
"""


#Getting the number of votes for bottom 90% of movies
m = movies['vote_count'].quantile(0.9)

#Average rating of all movies
C = movies['vote_average'].mean()

#Geting movies that have votes above threshold
filtered_movies = movies.copy().loc[movies['vote_count']>=m]

#Function to get IMDB weighted rating for movies
def weightrate(df, m=m, C=C):
    R = df['vote_average']
    v = df['vote_count']
    imbdb_weight = (v/(v+m))*R + (m/(v+m))*C
    return imbdb_weight

#filtering and showing most popular movies
filtered_movies['IMDB Rating'] = filtered_movies.apply(weightrate, axis=1)

print(filtered_movies.sort_values('IMDB Rating', ascending=False)[['title', 'IMDB Rating']])