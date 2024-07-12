import streamlit as st
import pandas as pd
import pickle

# Load your pre-trained RandomForestClassifier model
model = pickle.load(open("cancer_model1.pkl", "rb"))  # Assuming your model is saved as cancer_model.pkl

# Load the dataset and prepare the data
df = pd.read_csv('Cancer.csv')
df.drop('Patient Id', axis=1, inplace=True)
# Adjust the path as per your actual file location
X = df.drop('Level', axis=1)
y = df['Level']

# Function to predict cancer
def predict_cancer(new_data):
    predictions = model.predict(new_data)
    return predictions

# Streamlit app
st.title('Cancer Prediction App')

st.sidebar.header('Patient Data')

def user_input_features():
    age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=25)
    gender = st.sidebar.selectbox('Gender', options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')
    airpollution = st.sidebar.slider('Air Pollution', min_value=0, max_value=10, value=5)
    alcoholuse = st.sidebar.slider('Alcohol Use', min_value=0, max_value=10, value=5)
    dustallergy = st.sidebar.slider('Dust Allergy', min_value=0, max_value=10, value=5)
    occupationalhazards = st.sidebar.slider('Occupational Hazards', min_value=0, max_value=10, value=5)
    geneticrisk = st.sidebar.slider('Genetic Risk', min_value=0, max_value=10, value=5)
    chroniclungdisease = st.sidebar.slider('Chronic Lung Disease', min_value=0, max_value=10, value=5)
    balanceddiet = st.sidebar.slider('Balanced Diet', min_value=0, max_value=10, value=5)
    obesity = st.sidebar.slider('Obesity', min_value=0, max_value=10, value=5)
    smoking = st.sidebar.slider('Smoking', min_value=0, max_value=10, value=5)
    passivesmoker = st.sidebar.slider('Passive Smoker', min_value=0, max_value=10, value=5)
    chestpain = st.sidebar.slider('Chest Pain', min_value=0, max_value=10, value=5)
    coughingofblood = st.sidebar.slider('Coughing of Blood', min_value=0, max_value=10, value=5)
    fatigue = st.sidebar.slider('Fatigue', min_value=0, max_value=10, value=5)
    weightloss = st.sidebar.slider('Weight Loss', min_value=0, max_value=10, value=5)
    shortnessofbreath = st.sidebar.slider('Shortness of Breath', min_value=0, max_value=10, value=5)
    wheezing = st.sidebar.slider('Wheezing', min_value=0, max_value=10, value=5)
    swallowingdifficulty = st.sidebar.slider('Swallowing Difficulty', min_value=0, max_value=10, value=5)
    clubbingoffingernails = st.sidebar.slider('Clubbing of Finger Nails', min_value=0, max_value=10, value=5)
    frequentcold = st.sidebar.slider('Frequent Cold', min_value=0, max_value=10, value=5)
    drycough = st.sidebar.slider('Dry Cough', min_value=0, max_value=10, value=5)
    snoring = st.sidebar.slider('Snoring', min_value=0, max_value=10, value=5)

    data = {
        'Age': age,
        'Gender': gender,
        'AirPollution': airpollution,
        'Alcoholuse': alcoholuse,
        'DustAllergy': dustallergy,
        'OccuPationalHazards': occupationalhazards,
        'GeneticRisk': geneticrisk,
        'chronicLungDisease': chroniclungdisease,
        'BalancedDiet': balanceddiet,
        'Obesity': obesity,
        'Smoking': smoking,
        'PassiveSmoker': passivesmoker,
        'ChestPain': chestpain,
        'CoughingofBlood': coughingofblood,
        'Fatigue': fatigue,
        'WeightLoss': weightloss,
        'ShortnessofBreath': shortnessofbreath,
        'Wheezing': wheezing,
        'SwallowingDifficulty': swallowingdifficulty,
        'ClubbingofFingerNails': clubbingoffingernails,
        'FrequentCold': frequentcold,
        'DryCough': drycough,
        'Snoring': snoring,
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Predict using the model
predictions = predict_cancer(input_df)

# Display the result
st.subheader('Prediction')
result = 'The person is likely affected by cancer. Please consult a doctor for further evaluation.' if predictions[0] == 1 else 'The person is not likely affected by cancer.'
st.write(result)
