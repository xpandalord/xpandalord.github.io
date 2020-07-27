---
title: Predicting the outcome of a League of Legends Match
subtitle: How to decide based on a few factors
image: /assets/img/league-of-legends.jpg
---

# Dataset
Using the following dataset from Kaggle, [League of Legends Diamond Ranked Games (10 min)](https://www.kaggle.com/bobbyscience/league-of-legends-diamond-ranked-games-10-min), let's determine whether or not the blue team wins. The dataset gives us the number of wards placed and the number of wards destroyed by each team; which team obtained first blood; the number of kills, deaths, and assists of each team; the number of elite monsters, dragons, and heralds that were killed by each team; the number of towers destroyed by each team; and the amount of gold and experience obtained by each team along with the minions and jungle minions killed by each team.

There are a lot of redundant data including the features labeled as gameId, redFirstBlood, redDeaths, blueDeaths, blueGoldPerMin, redGoldPerMin, blueCSPerMin, redCSPerMin, redGoldDiff, and redExperienceDiff due to either being irrelavant to the outcome of already having other columns that give the same data and have the same significance.

## Target, Metric, and Baseline
Our target for this dataset is the blueWins feature telling us whether the blue team won with a 1, and otherwise a 0. Thus, we will be using logistic regression and random forest classifier from the sklearn library to fit this binary classification. This will allow us to obtain the accuracy of our model by scoring the model's predictions based on the actual values in our dataset. An easy way to judge whether our models are sufficient is by comparing their accuracy scores with a baseline prediction.

With this dataset, if we predict that blue always loses, we will be right half of the time. In other words our baseline model is predicting the blue team to always loses with an accuracy score of 0.50. The following is the code to find the majority class between the two possible outcomes and our model's accuracy score.
 
    # converting target into a list to find the majority class
    # using value_counts(normalize=True) to find the accuracy score
    y = y_train.values.tolist()
    majority_class = max(set(y), key = y.count)
    print(f'Majority Class: {majority_class}')
    print(f'Accuracy Score: {y_train.value_counts(normalize=True)[0]:,.2f}')

![League of Legends Majority Class and Its Accuracy Score](/assets/img/league-majority.png)