"""
Used making tensorflow a tensorflow moden and predicting prices on DBA.dk for Bang & Olufsen ads.

This was made for a school project about machine learning.
"""
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from fuzzywuzzy import process
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential, load_model
from sklearn import preprocessing


class machineLearning():
    """Handles all functions needed for machine learning."""

    def build_model(self):
        """Create the model for machine learning."""
        model = Sequential()
        model.add(Dense(300, activation='relu', input_dim=247))
        model.add(Dense(250, activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))
        opt = optimizers.Adam(epsilon=1.0)
        model.compile(loss='mse', optimizer=opt)  # This model seems to be the one that gives the most accurate results.

        model.save(os.path.join(sys.path[0], r'../data/model.h5'))

    def load_model(self):
        """Load the model. Create it and save to file if it doesn't exist."""
        if os.path.exists(os.path.join(sys.path[0], r'../data/model.h5')):
            return load_model(os.path.join(sys.path[0], r'../data/model.h5'))
        else:
            self.build_model()
            return load_model(os.path.join(sys.path[0], r'../data/model.h5'))

    def train_or_predict(self, iterations, predict=[]):
        """
        Use to train the model or to make a prediction.

        Supply an a list of ['model_modelName', 'condition_conditionName', 'watt'].
        """
        # Load dummified dataframe records create by DBA scrape
        # Tell user to scrape if file doesn't exist
        if os.path.exists(os.path.join(sys.path[0], r'../data/scraped_dataframe_records_dummified.json')) is not True and predict != []:
            print("No training data available - do a scrape to get it")
            time.sleep(3)
            return
        else:
            training_data = pd.read_json(os.path.join(sys.path[0], r'../data/scraped_dataframe_records_dummified.json'))
        model = self.load_model()

        x = pd.DataFrame(training_data.loc[:, training_data.columns != 'price']).values.tolist()  # list_without_price
        y = pd.DataFrame(training_data.loc[:, training_data.columns == 'price']).values.tolist()  # list_price_only

        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        x = pd.DataFrame(x_scaled).values.tolist()

        # Prediction part
        if predict != []:
            new_row = {predict[0]: float(1), predict[1]: float(1), 'watt': float(predict[2])}
            training_data = training_data.iloc[0:0]
            prediction_df = training_data.append(new_row, ignore_index=True)
            prediction_df = prediction_df.fillna(0)
            prediction_list = pd.DataFrame(prediction_df.loc[:, training_data.columns != 'price']).values.tolist()
            scaled_prediction = min_max_scaler.transform(prediction_list)
            return model.predict(pd.DataFrame(scaled_prediction).values.tolist())

        # Training part
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        history = model.fit(x, y, callbacks=[early_stop], epochs=iterations, verbose=2)
        lx = list(range(0, len(history.history['loss'])))
        ly = history.history['loss']
        plt.plot(lx, ly, c='red')
        model.save(os.path.join(sys.path[0], r'../data/model.h5'))
        print("Done training, close graph to return to main menu")
        plt.show()  # Show the training progress in a graph

    def predict(self):
        """To be called from "UI". Will handle user input for predictions."""
        print("##################################################################")
        print("####      Predict the price of a Bang & Olufsen product      #####")
        print("####         by supplying model, watt and condition          #####")
        print("##################################################################")

        model = input("Model: ")
        # Make sure the chosen model is available for use.
        # If it isn't, make a suggestion for the nearest available model.
        with open(os.path.join(sys.path[0], r'../data/allowed_models.json')) as json_file:
            allowed_models = json.load(json_file)
            if model not in allowed_models:
                closest_match = process.extract(model, allowed_models, limit=1)
                print(f"Model was not in the allowed model list, closest match is {closest_match[0][0]}")
                model_choise = input(f"Do you want to use {closest_match[0][0]} as model? (Y/N)")
                if model_choise.lower() == "y":
                    model = closest_match[0][0]
                else:
                    return
        condition = input("Condition(defekt, rimelig, god, perfekt): ")
        if condition.lower() not in ['defekt', 'rimelig', 'god', 'perfekt']:
            print("Condition not supported")
            return
        watt = input("Watt: ")
        if watt == "":
            watt = "0"

        prediction = self.train_or_predict(0, [f"model_{model}", f"condition_{condition}", watt])
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Predicted price of {model} in {condition} condition with {watt} watt is: {round(prediction[0][0])} DKK")
