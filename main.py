import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Code from Google colaboratory

# Reading data
animes = pd.read_csv('anime.csv')
# rating_of_anime = pd.read_csv('/content/rating.csv')
# animes.drop(['genre'], axis=1, inplace=True)
# animes.drop(['type'], axis=1, inplace=True)
# animes.drop(['episodes'], axis=1, inplace=True) # We drop the total rating as we need to find similar ratings between users
# animes.drop(['rating'], axis=1, inplace=True)
# animes.drop(['members'], axis=1, inplace=True)

# Making a matrix and add a mask to optimize AI's work
# user_anime_matrix = animes.pivot(index='anime_id', columns='user_id', values='ratings_of_anime')
# user_anime_matrix.fillna(0, inplace = True)
# user_votes = animes.groupby('user_id')['ratings_of_anime'].agg('count')
# anime_votes = animes.groupby('anime_id')['ratings_of_anime'].agg('count')
# users_mask = user_votes[user_votes > 50].index
# anime_mask = anime_votes[anime_votes > 30].index
# user_anime_matrix = user_anime_matrix.loc[anime_mask,:]
# user_anime_matrix = user_anime_matrix.loc[:,users_mask]

#Saving pivot to use it in Pycharm (connected with some errors while using pandas in pycharm)
#user_anime_matrix.to_csv('user_anime_matrix', sep=',')

# Reading prepared data from Google colaboratory
user_anime_matrix = pd.read_csv('user_anime_matrix')

# Making a csr matrix
csr_anime = csr_matrix(user_anime_matrix.values)

#Reseting the index for comfort searching
user_anime_matrix = user_anime_matrix.rename_axis(None, axis = 1).reset_index()

# Training the model
anime = NearestNeighbors(metric = 'cosine', algorithm = 'brute', n_neighbors = 20, n_jobs = -1)
anime.fit(csr_anime)

#Making a Telegram bot
import logging
from aiogram import Bot, Dispatcher, executor, types

TOKEN = 'token'
logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
  user_id = message.from_user.id
  user_name = message.from_user.first_name
  logging.info(f'{user_id=}, {user_name=}')
  await message.answer(f"Hi, {user_name}! I'm colaborative recomendation system. You can ask me what you can watch at your leisure and I will pick up an anime for you. Please, type /help to see list of commands " )

@dp.message_handler(commands=['anime'], state=None)
async def anime_handler(message: types.Message):
  await message.answer('Write the name of the anime, based on which I could give you a recommendation')
  @dp.message_handler()
  async def anime_rec(message: types.Message):
    search_word = message.text
    recommendations = 3
    anime_search = animes[animes['name'].str.contains(search_word)]
    Anime_ids = user_anime_matrix[user_anime_matrix['anime_id'].isin(anime_search['anime_id'])].index[0]
    distances, indices = anime.kneighbors(csr_anime[Anime_ids], n_neighbors = recommendations + 1)
    indices_list = indices.squeeze().tolist()
    distances_list = distances.squeeze().tolist()
    indices_distances = list(zip(indices_list, distances_list))
    indices_distances_sorted = sorted(indices_distances, key = lambda x: x[1], reverse = False)
    indices_distances_sorted = indices_distances_sorted[1:]
    recom_list = []
    for ind_dist in indices_distances_sorted:
      matrix_anime_id = user_anime_matrix.iloc[ind_dist[0]]['anime_id']
      id = animes[animes['anime_id'] == matrix_anime_id].index
      title = animes.iloc[id]['name'].values[0]
      recom_list.append(title)
    name1 = str(recom_list[0])
    name2 = str(recom_list[1])
    name3 = str(recom_list[2])
    await message.answer("You should try this titles:")
    await message.answer(name1)
    await message.answer(name2)
    await message.answer(name3)

HELP = """
/help - list of commands
/anime - give a recommendation on anime
"""

@dp.message_handler(commands=['help'])
async def movie_handler(message: types.Message):
  await message.answer(text = HELP)

if __name__ == '__main__':
    executor.start_polling(dp)