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

## Logistic Regression
For our logistic regression, we will import the model from sklearn.liner_model, as well as import make_pipeline from sklearn.pipeline, and standardscaler from sklearn.preprocessing. This enables us to create a pipeline so we can scale the data and fit to the model simultaneously. We obtained a training accuracy of 0.74 and a validation accuracy of 0.72, which is great compared to our baseline score of 0.50.

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Instantiate model
    log = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )

    # Fit model on training data
    log.fit(X_train, y_train)

    # Check performance metric (accuracy) on train, validation, and test sets
    print('Training Accuracy:', log.score(X_train, y_train))
    print('Validating Accuracy:', log.score(X_val, y_val))

![Logistic Regression Accuracy Scores](/assets/img/log-score.png)

## Logstic Regression ROC-AUC
Furthermore, we go ahead and obtain the Receiver Operating Characteristic (ROC) curve as well as the Area Under the ROC Curve (AUC). This allows us to measure how well our model is able to distringuish between classes. The higher the AUC is, the better the model is at predicting 0s as 0s and 1s as 1s. The ROC curve is plotted with the True Positive Rate (TPR) against the False Positive Rate (FPR), where the TPR is on the y-axis and the FPR is on the x-axis. 

    from sklearn.metrics import plot_roc_curve
    plot_roc_curve(log, X_test, y_test)

![Logistic Regression ROC Curve](/assets/img/log-roc-curve.png)

TPR is obtained by calculating 

![Actual Values Against Predicted Values](/assets/img/actual-predicted.png)