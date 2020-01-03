import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

#%matplotlib inline

from subprocess import check_output


user_col = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('C:/Users/SUP-IMUKHER/Desktop/Python+ Hadoop/Projects in Github/Collaborative Filtering/Movie Recomendation Engine/ml-100k/u.user', sep='|', names=user_col,encoding='latin-1', parse_dates=True)
rating_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('C:/Users/SUP-IMUKHER/Desktop/Python+ Hadoop/Projects in Github/Collaborative Filtering/Movie Recomendation Engine/ml-100k/u.data', sep='\t', names=rating_cols,encoding='latin-1') 
movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv('C:/Users/SUP-IMUKHER/Desktop/Python+ Hadoop/Projects in Github/Collaborative Filtering/Movie Recomendation Engine/ml-100k/u.item', sep='|', names=movie_cols, usecols=range(5),encoding='latin-1')
movie_ratings = pd.merge(movies, ratings)
df = pd.merge(movie_ratings, users)
df.drop(df.columns[[3,4,7]], axis=1, inplace=True)
ratings.drop( "unix_timestamp", inplace = True, axis = 1 ) 
movies.drop(movies.columns[[3,4]], inplace = True, axis = 1 )
movie_stats = df.groupby('title').agg({'rating': [np.size, np.mean]})
min_50 = movie_stats['rating']['size'] >= 50
movie_stats[min_50].sort_values([('rating', 'mean')], ascending=False).head()
#Raters' Age Distribution
users.age.plot.hist(bins=25)
plt.title("Distribution of Users' Ages")
plt.ylabel('Number of Users')
plt.xlabel('Age');
#Pivot Table
ratings_matrix = ratings.pivot_table(index=['movie_id'],columns=['user_id'],values='rating').reset_index(drop=True)
ratings_matrix.fillna( 0, inplace = True )
ratings_matrix.head()
movie_similarity = 1 - pairwise_distances( ratings_matrix.as_matrix(), metric="cosine" )
np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
ratings_matrix = pd.DataFrame( movie_similarity )
ratings_matrix.head(5)
try:
	user_inp = input("Enter movie Name :- ")
	inp=movies[movies['title']==user_inp].index.tolist()
	inp=inp[0]
	movies['similarity'] = ratings_matrix.iloc[inp]
	movies.columns = ['movie_id', 'title', 'release_date','similarity']
except:
	print("Sorry, the movie is not in the database!")
print("Recommended movies based on your choice of ",user_inp ,": \n", movies.sort_values( ["similarity"], ascending = False )[1:10])