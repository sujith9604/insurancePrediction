# Insurance Claims Prediction Project

This repository contains a machine learning project aimed at predicting whether a policyholder will file an insurance claim in the next 6 months or not. The dataset used for this project was provided by Analytics Vidhya as part of their hackathon on November 14th, 2022.

## Dataset

The dataset consists of three files: `train.csv` and `test.csv`. Below is a brief description of the variables included in the dataset:

- **policy_id**: Unique identifier of the policyholder
- **policy_tenure**: Time period of the policy
- **age_of_car**: Normalized age of the car in years
- **age_of_policyholder**: Normalized age of policyholder in years
- **area_cluster**: Area cluster of the policyholder
- **population_density**: Population density of the city (Policyholder City)
- **make**: Encoded Manufacturer/company of the car
- **segment**: Segment of the car (A/ B1/ B2/ C1/ C2)
- **model**: Encoded name of the car
- **fuel_type**: Type of fuel used by the car
- **max_torque**: Maximum Torque generated by the car (Nm@rpm)
- **max_power**: Maximum Power generated by the car (bhp@rpm)
- **engine_type**: Type of engine used in the car
- **airbags**: Number of airbags installed in the car
- **is_esc**: Boolean flag indicating whether Electronic Stability Control (ESC) is present in the car or not.
- **is_adjustable_steering**: Boolean flag indicating whether the steering wheel of the car is adjustable or not.
- **is_tpms**: Boolean flag indicating whether Tyre Pressure Monitoring System (TPMS) is present in the car or not.
- **is_parking_sensors**: Boolean flag indicating whether parking sensors are present in the car or not.
- **is_parking_camera**: Boolean flag indicating whether the parking camera is present in the car or not.
- **rear_brakes_type**: Type of brakes used in the rear of the car
- **displacement**: Engine displacement of the car (cc)
- **cylinder**: Number of cylinders present in the engine of the car
- **transmission_type**: Transmission type of the car
- **gear_box**: Number of gears in the car
- **steering_type**: Type of the power steering present in the car
- **turning_radius**: The space a vehicle needs to make a certain turn (Meters)
- **length**: Length of the car (Millimeter)
- **width**: Width of the car (Millimeter)
- **height**: Height of the car (Millimeter)
- **gross_weight**: The maximum allowable weight of the fully-loaded car, including passengers, cargo, and equipment (Kg)
- **is_front_fog_lights**: Boolean flag indicating whether front fog lights are available in the car or not.
- **is_rear_window_wiper**: Boolean flag indicating whether the rear window wiper is available in the car or not.
- **is_rear_window_washer**: Boolean flag indicating whether the rear window washer is available in the car or not.
- **is_rear_window_defogger**: Boolean flag indicating whether rear window defogger is available in the car or not.
- **is_brake_assist**: Boolean flag indicating whether the brake assistance feature is available in the car or not.
- **is_power_door_lock**: Boolean flag indicating whether a power door lock is available in the car or not.
- **is_central_locking**: Boolean flag indicating whether the central locking feature is available in the car or not.
- **is_power_steering**: Boolean flag indicating whether power steering is available in the car or not.
- **is_driver_seat_height_adjustable**: Boolean flag indicating whether the height of the driver seat is adjustable or not.
- **is_day_night_rear_view_mirror**: Boolean flag indicating whether day & night rearview mirror is present in the car or not.
- **is_ecw**: Boolean flag indicating whether Engine Check Warning (ECW) is available in the car or not.
- **is_speed_alert**: Boolean flag indicating whether the speed alert system is available in the car or not.
- **ncap_rating**: Safety rating given by NCAP (out of 5)
- **is_claim**: Outcome: Boolean flag indicating whether the policyholder filed a claim in the next 6 months or not.

## Implementation

The project implements two classification algorithms, Logistic Regression and Decision Tree, using Scikit-Learn. Below is a summary of the implementation for each algorithm:

### Decision Tree

The Decision Tree algorithm is implemented using a custom approach. Functions for training the decision tree, making predictions, and calculating accuracy are defined. Additionally, k-fold cross-validation is implemented to evaluate the model's performance.

### Logistic Regression

Logistic Regression is implemented using functions for sigmoid activation, cost function, gradient descent, and accuracy calculation. Similar to Decision Tree, k-fold cross-validation is used for evaluation.

## Running the Code

To run the code, ensure you have Python installed along with the necessary libraries such as pandas, numpy, matplotlib, and Scikit-Learn. Then, simply execute the provided scripts `decision_tree.py` and `logistic_regression.py`. These can be installed as follows: 

   ```bash
   pip install pandas numpy matplotlib scikit-learn

   ```

## Results

The results, including accuracy scores for different values of k in k-fold cross-validation, are saved in text files (`output_dec.txt` for Decision Tree and `output_file.txt` for Logistic Regression). Additionally, plots showing the mean accuracy for different values of k are generated.

## Conclusion

This project demonstrates the implementation of two classification algorithms, Logistic Regression, and Decision Tree, for predicting insurance claims based on policyholder information. The results obtained through k-fold cross-validation provide insights into the performance of each algorithm on the given dataset.
