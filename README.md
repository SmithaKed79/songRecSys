**Objective**
The primary goal of this project was to build a personalized song recommendation system capable of predicting user ratings based on a combination of user-song interaction data and song metadata such as artist name, album, and title.

**Data Preprocessing**
The project utilized three datasets: train.csv, test.csv, and song_data.csv. Initially, I worked only with train.csv, focusing on user rating entries. For the early models, I performed basic cleaning, checking for duplicates and missing values, which were minimal and did not require significant handling. For XGboost, I incorporated song metadata from song_data.csv. I applied LabelEncoder() to convert categorical fields such as artist_name and release into numerical values. These encoded fields were then merged with the training dataset using song_id as the common key between the datasets. The best performing algorithm required minimal preprocessing, since it did not require encoding categorical features, but it did require replacing Nan values with string such as ‘missing’ to be able to form Pools.

**Feature Engineering**
From the merged dataset, I created two separate DataFrames:
    • X, containing features: ['user_id', 'song_id', 'artist_name', 'release', 'year'].
    • Y, consisting solely of the rating column as the target variable.
I then split the data into training and validation sets with an 80:20 ratio. Since CatBoost requires input in the form of Pool objects for efficient handling of categorical features, I converted the data into Pools before training the model using CatBoostRegressor.

**Evolution of training model**
    • Initial Attempt with SVD (Singular Value Decomposition):
Inspired by matrix factorization for collaborative filtering using user item utility matrix, I Started with collaborative filtering using the SVD algorithm from the Surprise library.  The model was trained solely on user_id, song_id, and rating, capturing latent factors for users and songs. I used RMSE for performance evaluation; while results were promising, SVD was limited to known users and items, with no way to incorporate song metadata. I assumed possible performance drawback to be no involvement of song metadata.
    • Experimented with XGBoost:
To include metadata in the model, I introduced XGBoost Regressor to incorporate song metadata such as artist_name, album, and title alongside user_id and song_id.  This Required manual encoding of categorical variables, adding preprocessing complexity. The model captured metadata relationships well and thus improving the RMSE(1.7273).

    • Transition to CatBoost:
CatBoost is a gradient boosting library developed by Yandex [1], designed to handle categorical features natively and reduce overfitting. It is particularly powerful for structured/tabular data where many features are non-numerical. It is considered to be a great choice for recommender systems, fraud detection, and any problem where relationships in categorical data are important. 
model = CatBoostRegressor(
	iterations=100,
	learning_rate=0.1,
	depth=6,
	eval_metric='RMSE',
	verbose=0)
<iterations>: Number of boosting rounds (trees).
<learning_rate>: Step size shrinkage used in updates.
<depth>: Depth of the individual trees.
<eval_metric>: Metric used for validation.
		<verbose>: Controls how often catBoost prints logs during training[2].
Although the xgboost regressor gave a good performance. I wanted to experiment with model with a better performance. The catBoost regressor was one of the algorithms suggested by OpenAI's ChatGPT. So I switched the model to CatBoost Regressor. Trained the model using features: user_id, song_id, year, release, and artist_name. Then built a prediction pipeline using CatBoost Pool, greatly simplifying the process of working with categorical data and required minimal preprocessing. It ended up having RMSE(1.6793).

**References:**
[1]  A. V. Dorogush, V. Ershov, and A. Gulin, "CatBoost: gradient boosting with categorical features support," Workshop on Machine Learning Systems at NIPS 2017, Long Beach, CA, USA, Dec. 2017. [Online]. Available: http://learningsys.org/nips17/assets/papers/paper_11.pdf​
[2] CatBoost Developers, "CatBoostRegressor," CatBoost Documentation, [Online]. Available: https://catboost.ai/docs/en/concepts/python-reference_catboostregressor. [Accessed: Apr. 13, 2025].
