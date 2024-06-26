import pandas as pd
from sklearn.model_selection import train_test_split
from frontend.modelling import Modelling

def predict_failure(classifier, label_encoder, X, y):
    print("\nAll Models have trained.\nPress 1 to predict new machine failure type or\nPress 2 to test a certain percentage of data.\nEnter any other key to exit the menu.")
    choice = input("Enter your choice: ").strip()
    
    if choice == "1":
        print("\nEnter the details of the new machine to predict its failure type:")
        air_temp = float(input("Air temperature [K]: "))
        process_temp = float(input("Process temperature [K]: "))
        rotational_speed = float(input("Rotational speed [rpm]: "))
        torque = float(input("Torque [Nm]: "))
        tool_wear = float(input("Tool wear [min]: "))
        machine_type = input("Machine Type (L, M, H): ")

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

        prediction = classifier.predict(new_data)

        if prediction[0] == 1:
            print("\nPredicted Failure Type: Failure")
        else:
            print("\nPredicted Failure Type: Non-failure")
    elif choice == "2":
        test_size = get_test_size()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        classification = Modelling(X_train, y_train, X_test, y_test, [classifier])
        classification.fit()
        classification.results()
        print('Accuracy of model:', classification.best_model_accuracy())
        print('Training Runtime in seconds', classification.best_model_runtime())
    else: 
        print("Exiting without testing a new machine.")

def get_test_size():
    while True:
        try:
            test_size = float(input("Enter the percentage of data to use for testing (e.g., 10 for 10%): ")) / 100
            if 0 < test_size < 1 < 100:
                return test_size
            else:
                print("Please enter a valid percentage between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")
