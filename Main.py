from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt

# players we are looking for predictions on, use full name, not case-sensitive
people = []
# players who are on the home team
home_players = []

# list that will contain the player ids
ids = []

# dictionary used to determine which team a player is playing against
team = {'home': 'NOP'
         , 'away': 'HOU'}

match_up = {}
# pts = -3, ast = -8, reb = -9
stat = -3

#get home team from input

#get away team from input

#get stat from input

####
#Make get players from team a function
####

#get players on the home team
h_team = teams.find_team_by_abbreviation(abbreviation= team['home'])
h_id = h_team['id']
h_roster = commonteamroster.CommonTeamRoster(team_id= h_id)
dfh = h_roster.get_dict()
#add the players on the home team to list of people
for item in dfh['resultSets'][0]['rowSet']:
    people.append(item[3])
    home_players.append(item[3])

#get players on the away team
a_team = teams.find_team_by_abbreviation(abbreviation= team['away'])
a_id = a_team['id']
a_roster = commonteamroster.CommonTeamRoster(team_id= a_id)
dfa = a_roster.get_dict()
#add the players on the away team to list of people
for item in dfa['resultSets'][0]['rowSet']:
    people.append(item[3])

# get the player id from full name
print(people)
for person in people:
    # get the player id from their full name
    player = players.find_players_by_full_name(person)
    person = person.lower()
    #print(player)
    try:
        ids.append(player[0]['id'])
    except:
        continue
    # determine the team a player is playing against based on weather they are home or away
    if person in home_players:
        match_up[person] = team['away']
    else:
        match_up[person] = team['home']

#print(ids)
#print(player[0].keys())
#print(match_up)

# loop through each pid and make prediction
# Make prediction a function
for pid in ids:

    name = players.find_player_by_id(pid)
    lower_name = name['full_name'].lower()

    # get the player game log using the nba api
    games = playergamelog.PlayerGameLog( player_id=str(pid), season='2023-24' )
    # create data frame and dictionary from the game log
    p = games.get_data_frames()
    test = games.get_dict()


    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    #print(p)


    #get the data we will use
    i = 0
    j = 0
    avg_points = 0
    points = []
    match_up_points = 0
    home = []
    big_dict = dict()
    for row in test['resultSets'][0]['rowSet']:
        # calculate the player avg ppg
        if test['resultSets'][0]['rowSet'][i][5] != None:
            points.append(test['resultSets'][0]['rowSet'][i][stat])
            avg_points += test['resultSets'][0]['rowSet'][i][stat]

            # calculate the players avg points against this team
            if match_up[lower_name] in test['resultSets'][0]['rowSet'][i][4]:
                match_up_points += test['resultSets'][0]['rowSet'][i][stat]
                j += 1

            # Determine weather home or away, 0 is away, home is 1
            if '@' in test['resultSets'][0]['rowSet'][i][4]:
                home.append(0)
            else:
                home.append(1)

        i+=1

    if i != 0:
        avg_points = avg_points / i
    else:
        avg_points = 0
    #if player has never played team set matchup points to average
    if j != 0:
        match_up_points = match_up_points/j
    else:
        match_up_points = avg_points
    #print(points)
    #print(avg)
    if len(player) == 0:
        continue

    #################
    # prediction model for pts, ast, reb
    #
    #
    ###################

    # create our data set
    data = {'points': points
            , 'average_points': avg_points
            , 'home_or_away': home
            , 'match_up_points': match_up_points}
    df = pd.DataFrame(data)
    # do not make a prediction on players who have played less then 10 games
    if len(df) < 10:
        continue
    # Reverse the order of the DataFrame so the most recent game is first in the list
    df = df[::-1].reset_index(drop=True)

    # Create a new column 'next_game_points' shifted by one to represent the target variable
    df['next_game_points'] = df['points'].shift(-1)
    # Drop the last row with NaN in 'next_game_points' as there's no information about the next game
    df = df.dropna()
    #print(df)

    # Features and target variable
    X = df[['points', 'average_points', 'home_or_away', 'match_up_points']]
    y = df['next_game_points']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Create a Linear Regression model
    linear_model = LinearRegression()
    # Train the model
    linear_model.fit(X_train, y_train)
    # Make predictions on the test set
    predictions_linear = linear_model.predict(X_test)


    # Create Random Forest Regressor model
    rf_model = RandomForestRegressor()
    # Train the model
    rf_model.fit(X_train, y_train)
    # Make predictions on the test set
    predictions_rf = rf_model.predict(X_test)

    # Evaluate models
    mae_linear = mean_absolute_error(y_test, predictions_linear)
    mae_rf = mean_absolute_error(y_test, predictions_rf)

    # Cross validate the model
    # Combine the features and target variable into one DataFrame
    cross_data = pd.concat([X, y], axis=1)

    # Define the number of folds for cross-validation
    num_folds = 5 # You can adjust the number of folds based on your preference

    # Set up KFold for cross-validation
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Perform cross-validation for Linear Regression
    linear_mae_cv = cross_val_score(linear_model, X, y, cv=kf, scoring=make_scorer(mean_absolute_error))
    # Perform cross-validation for Random Forest Regressor
    rf_mae_cv = cross_val_score(rf_model, X, y, cv=kf, scoring=make_scorer(mean_absolute_error))

    # Calculate the average cross-validation MAE for each model
    avg_linear_mae_cv = linear_mae_cv.mean()
    avg_rf_mae_cv = rf_mae_cv.mean()
    combined_cv_result = (avg_linear_mae_cv + avg_rf_mae_cv) / 2

    # Combine predictions using model averaging
    combined_predictions = (predictions_linear + predictions_rf) / 2
    # Evaluate the combined predictions
    mae_combined = mean_absolute_error(y_test, combined_predictions)

    # print the players name
    print(name['full_name'])
    # print points list and average
    print(f'Points in past games: {points}')
    print(f'Avg PPG: {avg_points}')
    print(f'Avg Match-up PPG: {match_up_points}')

    # Now, use the trained model to predict the points for the next game
    # Given the last known points and player's average points:
    last_known_points = points[0]
    print(last_known_points)
    player_average = df['average_points'].iloc[-1:]
    home = 0
    if lower_name in home_players:
        home = 1
    new_data = pd.DataFrame({'points': last_known_points, 'average_points': player_average,
                             'home_or_away': home, 'match_up_points': match_up_points})
    predicted_points_linear = linear_model.predict(new_data)
    predicted_points_rf = rf_model.predict(new_data)
    predicted_points_combined = (linear_model.predict(new_data) + rf_model.predict(new_data)) / 2

    # Print results
    print('\nLinear regression predictions:')
    print(f'Linear Regression - Mean Absolute Error: {mae_linear}')
    print(f'Linear Regression - Cross-Validation Mean Absolute Error: {linear_mae_cv.mean()}')
    print(f'Linear Predicted Points for the Next Game: {predicted_points_linear[0]}')
    print('\nRandom Forest Regression Predictions:')
    print(f'Random Forest Regression - Mean Absolute Error: {mae_rf}')
    print(f'Random Forest Regression - Cross-Validation Mean Absolute Error: {rf_mae_cv.mean()}')
    print(f'Random Forest predicted Points for the Next Game: {predicted_points_rf[0]}')
    print('\nCombined Predictions:')
    print(f'Combined Predictions - Mean Absolute Error: {mae_combined}')
    print(f'Combined Cross-Validation Result: {combined_cv_result}')
    print(f'Combined Predicted Points for the Next Game: {predicted_points_combined[0]}')
    print('***********************************************************************************\n')

    # Create scatter plot to display accuracy
    plot = False
    if plot:
        # Scatter plot predicted vs actual
        plt.scatter(predictions_linear, y_test, label='Linear Regression', color='blue')  # Swap x and y here
        plt.scatter(predictions_rf, y_test, label='Random Forest Regression', color='green')  # Swap x and y here
        plt.scatter(combined_predictions, y_test, label='Combined Predictions', color='orange')  # Swap x and y here

        # Add line of best fit
        linear_fit = np.polyfit(predictions_linear, y_test, 1)  # Swap x and y here
        rf_fit = np.polyfit(predictions_rf, y_test, 1)  # Swap x and y here
        combined_fit = np.polyfit(combined_predictions, y_test, 1)  # Swap x and y here

        # Add diagonal line for reference (ideal prediction)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red',
             label='Ideal Prediction')

        # Set labels and title
        plt.xlabel('Predicted Points')
        plt.ylabel('Actual Points')
        plt.title(name['full_name'])

        # Add legend
        plt.legend()

        # Display the plot
        plt.show()

