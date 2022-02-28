from utils import *
import numpy as np
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: # Accuracy
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
#helper function to get the predicted_positive_rate

def compare_probs(prob1, prob2, epsilon):    
    return abs(prob1 - prob2) <= epsilon
    
def get_num_predicted_positives_rate(data):
    return get_num_predicted_positives(data)/ len(data)

def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}
    # Must complete this function!
    return get_optimal_threshold(categorical_results, epsilon, get_num_predicted_positives_rate)

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""


def enforce_equal_opportunity(categorical_results, epsilon):

    thresholds = {}
    equal_opportunity_data = {}

    # Must complete this function!

    return get_optimal_threshold(categorical_results, epsilon, get_true_positive_rate)

    #return None, None

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}

    # Must complete this function!

    threshold_value = np.arange(0.0, 1.0, 0.01)
    thresholds = dict.fromkeys(categorical_results, 0)

    for group in categorical_results.keys():
        max_accuracy = 0
        max_accuracy_threshold = 0
        for val in threshold_value:
            temp_data = apply_threshold(categorical_results[group], val)
            accuracy = get_num_correct(temp_data)/len(temp_data)
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                max_accuracy_threshold = val
        thresholds[group] = max_accuracy_threshold


    for group in categorical_results.keys():
        mp_data[group] = apply_threshold(categorical_results[group],thresholds[group])

    return mp_data, thresholds
    #return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}

    return get_optimal_threshold(categorical_results, epsilon, get_positive_predictive_value)
    #return None, None

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    test = {}
    single_threshold_data = {}

    threshold_values = np.arange(0.0, 1.0, 0.01)

    max_accuracy = 0
    threshold = 0
    temp_data = {}

    for x in threshold_values:
        for group in categorical_results.keys():
            temp_data[group] = apply_threshold(categorical_results[group], x)
        total_accuracy = get_total_accuracy(temp_data)
        if total_accuracy > max_accuracy:
            max_accuracy = total_accuracy
            threshold = x
    thresholds = dict.fromkeys(categorical_results, threshold)


    for k in categorical_results.keys():
        single_threshold_data[k] = apply_threshold(categorical_results[k], threshold)

    # Must complete this function!
    return single_threshold_data, thresholds

def similar_threshold_values(dict_data, epsilon):
    keys = list(dict_data.keys())
    final = []
    found_possible_threshold = False

    for value1 in dict_data[keys[0]]:
        found_possible_threshold = False
        for value2 in dict_data[keys[1]]:
            if found_possible_threshold:
                continue
            if compare_probs(value1[0], value2[0], epsilon):
                for value3 in dict_data[keys[2]]:
                    if found_possible_threshold:
                        continue
                    if compare_probs(value1[0], value3[0], epsilon) and compare_probs(value2[0], value3[0], epsilon):
                        for value4 in dict_data[keys[3]]:
                            if found_possible_threshold:
                                continue
                            if compare_probs(value1[0], value4[0], epsilon) and compare_probs(value2[0], value4[0], epsilon) and compare_probs(value3[0], value4[0], epsilon):
                                final.append([value1, value2, value3, value4])
                                #print(str(value1) + " " + str(value2) + " " + str(value3) + " " + str(value4) + " " )
                                found_possible_threshold = True





    return final

def get_optimal_threshold(categorical_results, epsilon, method):
    thresholds = {}
    threshold_data = {}

    threshold_values = np.arange(0.0, 1.01, 0.01)

    group_rate_threshold = {}
    for threshold in threshold_values:
        for group, group_data in categorical_results.items():
            result_data = apply_threshold(group_data, threshold)
            prob = method(result_data)
            if group not in group_rate_threshold:
                group_rate_threshold[group] = []
            group_rate_threshold[group].append([prob, threshold])

    threshold_list = similar_threshold_values(group_rate_threshold, epsilon)

    classification = {}

    max_acc = 0

    for threshold in threshold_list:
        classification[list(categorical_results.keys())[0]] = apply_threshold(list(categorical_results.values())[0], threshold[0][1])
        classification[list(categorical_results.keys())[1]] = apply_threshold(list(categorical_results.values())[1], threshold[1][1])
        classification[list(categorical_results.keys())[2]] = apply_threshold(list(categorical_results.values())[2], threshold[2][1])
        classification[list(categorical_results.keys())[3]] = apply_threshold(list(categorical_results.values())[3], threshold[3][1])
            

        acc = get_total_accuracy(classification)

        if acc > max_acc:
            max_acc = acc
            thresholds[list(categorical_results.keys())[0]] = threshold[0][1]
            thresholds[list(categorical_results.keys())[1]] = threshold[1][1]
            thresholds[list(categorical_results.keys())[2]] = threshold[2][1]
            thresholds[list(categorical_results.keys())[3]] = threshold[3][1]


    for group in categorical_results.keys():
        threshold_data[group] = apply_threshold(categorical_results[group], thresholds[group])
    return threshold_data, thresholds