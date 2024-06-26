import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from modelling import Modelling

st.set_page_config(page_title="Machine Failure Prediction", layout="centered")

data = pd.read_csv('machine_failure_prediction.csv')
data.head()
data = data.drop(["UDI", 'Product ID'], axis=1)
data['nf'] = data['Tool wear [min]'] * data['Torque [Nm]']

label_encoder = LabelEncoder()
label_encoder.fit(data['Type'])
data['Type'] = label_encoder.transform(data['Type'])

X = data.drop(['Failure Type', 'Target'], axis=1)
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=1)

def train_model():
    classification = Modelling(X_train, y_train, X_test, y_test, [knn])
    classification.fit()
    classification.results()
    return classification

def main():
    st.markdown("<h1 style='text-align: center;'>Machine Failure Prediction</h1>", unsafe_allow_html=True)
    
    if 'classification' not in st.session_state:
       st.session_state.classification = None

    if 'training_disabled' not in st.session_state:
        st.session_state.training_disabled = False


    buttonText = 'Model is already trained' if st.session_state.training_disabled else 'Start Training My Model'

    if st.button(buttonText, use_container_width=True, disabled=st.session_state.training_disabled):
        if not st.session_state.training_disabled:
            st.write("Training models...")
            st.session_state.classification = train_model()
            st.write('Accuracy of model:', st.session_state.classification.best_model_accuracy())
            st.write('Training Runtime in seconds:', st.session_state.classification.best_model_runtime())
            st.session_state.training_disabled = True

    if st.session_state.classification is not None:
        with st.form(key='predict_form'):
            st.subheader("Enter details of the new machine:")
            air_temp = st.number_input("Air temperature [K]", min_value=0.0, step=1.0)
            process_temp = st.number_input("Process temperature [K]", min_value=0.0, step=1.0)
            rotational_speed = st.number_input("Rotational speed [rpm]", min_value=0.0, step=1.0)
            torque = st.number_input("Torque [Nm]", min_value=0.0, step=1.0)
            tool_wear = st.number_input("Tool wear [min]", min_value=0.0, step=1.0)
            machine_type = st.selectbox("Machine Type", ['L', 'M', 'H'])
            submit_button = st.form_submit_button(label='Predict Failure for New Machine')

            if submit_button:
                new_data = pd.DataFrame({
                    'Type': [machine_type],
                    'Air temperature [K]': [air_temp],
                    'Process temperature [K]': [process_temp],
                    'Rotational speed [rpm]': [rotational_speed],
                    'Torque [Nm]': [torque],
                    'Tool wear [min]': [tool_wear],
                })

                new_data['Type'] = label_encoder.transform(new_data['Type'])
                new_data['nf'] = new_data['Tool wear [min]'] * new_data['Torque [Nm]']

                prediction = st.session_state.classification.models[0].predict(new_data)

                if prediction[0] == 1:
                    st.write("Predicted Failure Type: Failure")
                else:
                    st.write("Predicted Failure Type: Non-failure")

        with st.form(key='test_form'):
            test_size = st.slider("Enter the percentage of data to use for testing:", min_value=0, max_value=100) / 100.0
            test_submit_button = st.form_submit_button(label='Test Data on Certain Percentage')

            if test_submit_button:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                classification = Modelling(X_train, y_train, X_test, y_test, [KNeighborsClassifier(n_neighbors=1)])
                classification.fit()
                classification.results()

                st.write('Accuracy of model:', classification.best_model_accuracy())
                st.write('Training Runtime in seconds:', classification.best_model_runtime())

if __name__ == '__main__':
    main()