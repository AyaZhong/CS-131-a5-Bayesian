import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

#============================================================
# Read speed and likelihood data files
likelihood_file = "likelihood.txt"
training_file = "training.txt"
test_file = "testing.txt"
#=====================================

# Read likelihood data
likelihood_data = np.genfromtxt(likelihood_file, delimiter=' ')
#=======================================

# Process training data
training_data = np.genfromtxt(training_file, delimiter=' ')
# Handle NaN values
median_training_data = np.nanmedian(training_data, axis=0)
# Replace NaN with median in the training data
training_data_with_median = np.where(np.isnan(training_data), median_training_data, training_data)

# Split training data into 10 sets of bird velocity and 10 sets of airplane velocity
time_velocity_bird = training_data_with_median[:10]
time_velocity_airplane = training_data_with_median[10:]

#============use extra feature, training std===============
def traningdata_std_each_row_element(time_velocity):
    stds = np.std(time_velocity)
    return stds

def traningdata_std_each_row(time_velocity):
    sum_stds = 0
    numbers_of_row = len(time_velocity)
    for row in time_velocity:
        each_row_std = traningdata_std_each_row_element(row)
        sum_stds += each_row_std

    avg_std = sum_stds / numbers_of_row
    return avg_std

# Calculate standard deviation for each row in bird and airplane training data
training_bird_std = traningdata_std_each_row(time_velocity_bird)
training_airplane_std = traningdata_std_each_row(time_velocity_airplane)

#============================================

# Initialize probabilities
initial_prior_bird = 0.5
initial_prior_airplane = 0.5

# Transition probabilities
transition_bird_bird_same_state = 0.9
transition_airplane_bird_different_state = 0.1
transition_airplane_airplane_same_state = 0.9
transition_bird_airplane_different_state = 0.1
#=================================================

def Naive_Recursive_Bayesian_Testing(test_file):
    # =====================
    # Prepare test data
    test_data = np.genfromtxt(test_file, delimiter=' ')
    # Handle NaN values
    median_test_data = np.nanmedian(test_data, axis=0)

    # Replace NaN with median in the test data
    test_data_with_median = np.where(np.isnan(test_data), median_test_data, test_data)

    #  Choose likelihood probabilities corresponding to velocity
    likelihood_bird = likelihood_data[0]
    likelihood_airplane = likelihood_data[1]

    # Use interpolation function
    interp_likelihood_bird = interp1d(np.linspace(0, 200, len(likelihood_bird)), likelihood_bird, kind='linear',
                                      fill_value="extrapolate")
    interp_likelihood_airplane = interp1d(np.linspace(0, 200, len(likelihood_airplane)), likelihood_airplane,
                                          kind='linear', fill_value="extrapolate")

    # Iterate to calculate probabilities for each velocity point
    for i in range(len(test_data_with_median)):
        # Calculate standard deviation for each test sample
        test_stds = np.std(test_data_with_median[i])

        # Calculate the difference between the standard deviation of test data and training data for airplane and bird
        diff_bird_std = abs(test_stds - training_bird_std)
        diff_airplane_std = abs(test_stds - training_airplane_std)

        # # Print differences
        # print("Difference between test sample and airplane standard deviation", diff_airplane_std)
        # print("Difference between test sample and bird standard deviation", diff_bird_std)

        # Adjust prior probabilities
        if diff_bird_std < diff_airplane_std:
            # If the standard deviation of the test data is closer to bird, increase the prior probability of bird
            adjusted_prior_bird = initial_prior_bird * (1 + (1 - diff_bird_std / (diff_bird_std + diff_airplane_std)))
            adjusted_prior_airplane = initial_prior_airplane
        else:

            adjusted_prior_airplane = initial_prior_airplane * (
                        1 + (1 - diff_airplane_std / (diff_bird_std + diff_airplane_std)))
            adjusted_prior_bird = initial_prior_bird

        # Use adjusted prior probabilities for the first iteration
        speed1 = test_data_with_median[i][0]
        likelihood_bird1 = interp_likelihood_bird(speed1)
        likelihood_airplane1 = interp_likelihood_airplane(speed1)

        p_bird1 = likelihood_bird1 * adjusted_prior_bird
        p_airplane1 = likelihood_airplane1 * adjusted_prior_airplane
        p_bird_update = p_bird1
        p_airplane_update = p_airplane1

        for j in range(1, len(test_data_with_median[i])):
            speed = test_data_with_median[i, j]
            likelihood_bird = interp_likelihood_bird(speed)
            likelihood_airplane = interp_likelihood_airplane(speed)

            # Update probabilities for each velocity point
            p_bird = likelihood_bird * (
                    p_bird_update * transition_bird_bird_same_state + p_airplane_update * transition_airplane_bird_different_state)
            p_airplane = likelihood_airplane * (
                    p_airplane_update * transition_airplane_airplane_same_state + p_bird_update * transition_bird_airplane_different_state)

            # Normalize probabilities
            total_probability = p_bird + p_airplane
            epsilon = 1e-10
            p_bird_normalized = p_bird / (total_probability + epsilon)
            p_airplane_normalized = p_airplane / (total_probability + epsilon)

            # Update variables for the next iteration
            p_bird_update = p_bird_normalized
            p_airplane_update = p_airplane_normalized

        # Print the final probabilities for each row
        # print(f"Object {i + 1} Final Probability - p(Bird): {p_bird_update}, p(Airplane): {p_airplane_update}")
        final_classification = 'Bird' if p_bird_update > p_airplane_update else 'Airplane'
        print(f"Object {i + 1} is classified as {final_classification}")

if __name__ == '__main__':
    Naive_Recursive_Bayesian_Testing(test_file)
