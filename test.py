
import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import streamlit
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
import streamlit as st




class KnowledgeBase:
    def __init__(self, data_path1, data_path2, data_path3):
        data1 = pd.read_csv(data_path1)
        data2 = pd.read_csv(data_path2)
        data3 = pd.read_csv(data_path3)
        data1 = data1.replace("t", 0)
        data1 = data1.replace("t'", 0)
        
        
        self.data = pd.concat([data1, data2, data3], ignore_index=True)

        self.data['Carbs'] = SimpleImputer(strategy='median').fit_transform(self.data[['Carbs']])

        self.data['Carbs'] = pd.to_numeric(self.data['Carbs'], errors='coerce')

    def get_food_info(self, food_id):
        return self.data.iloc[food_id]



class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((3, num_actions))  # 3 rows for breakfast, lunch, dinner

    def choose_action(self, meal_type):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_table[meal_type])

    def update_q_table(self, meal_type, action, reward, next_max_q_value):
        self.q_table[meal_type, action] += self.learning_rate * (reward + self.discount_factor * next_max_q_value - self.q_table[meal_type, action])




def load_model_and_data():
    knowledge_base = KnowledgeBase(r"C:\datasets\nutrients_csvfile.csv", 
                                    r"C:\datasets\DOHMH_MenuStat__Historical_.csv", 
                                    r"C:\datasets\fastfood.csv")
    num_food_items = len(knowledge_base.data)
    agent = QLearningAgent(num_food_items)
    num_episodes = 100
    meal_types = {'breakfast': 0, 'lunch': 1, 'dinner': 2}

    for episode in range(num_episodes):
        for meal_type, index in meal_types.items():
            action = agent.choose_action(index)
            food_info = knowledge_base.get_food_info(action)
            carbs_threshold = 50 # My test threshold in grams
            reward = -1 if food_info['Carbs'] > carbs_threshold else 1
            next_max_q_value = np.max(agent.q_table[index])
            agent.update_q_table(index, action, reward, next_max_q_value)
    
    best_choices = {meal: knowledge_base.get_food_info(np.argmax(agent.q_table[index]))['Food'] 
                    for meal, index in meal_types.items()}
    return best_choices

def main():
    st.title('Meal Recommendation System for Diabetic Patients')
    st.write("This app recommends daily meals for diabetic patients based on nutritional content.")

    if st.button('Get Daily Meal Recommendations'):
        best_choices = load_model_and_data()
        st.subheader("Today's Meal Plan:")
        for meal, food in best_choices.items():
            st.write(f"{meal.title()}: {food}")
    else:
        st.write("Click the button above to get today's meal recommendations.")

if __name__ == "__main__":
    st.set_page_config(page_title='Diabetic Meal Recommender', layout='wide')
    main()