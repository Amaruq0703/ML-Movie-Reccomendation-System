import pandas as pd
import sklearn.feature_extraction.text as sk
import sklearn.metrics.pairwise as skpair



movies = pd.read_csv('data/movies.csv')

#Assigning Term frequency inverse document ID
tfidf = sk.TfidfVectorizer(stop_words='english')

#Filling null values
movies['overview'] = movies['overview'].fillna('')

#Making a matrix of movie tfids
tfidf_matrix = tfidf.fit_transform(movies['overview'])

movie_tfids = pd.DataFrame(tfidf_matrix.toarray(), columns = tfidf.get_feature_names_out())

# Making a linear kernel matrix of tfid matrixes at each axis
movie_similarity_matrix = skpair.linear_kernel(tfidf_matrix, tfidf_matrix)

def similar_movies(title, nrmovies):
    idx = movies.loc[movies['title'] == title].index[0]
    scores = list(enumerate(movie_similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x:x[1],reverse=True)
    movieindices = [ind[0] for ind in scores][1:nrmovies+1]
    return movieindices

title = input('Search for movies like: ')
nrmovies = int(input('How many recommendations do you want?: '))

movieindices = similar_movies(title, nrmovies)

print(movies['title'].iloc[movieindices])


