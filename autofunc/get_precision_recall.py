"""

Find the accuracy of a prediction based on the known results in a testing set and the predicted results from
a training set. The accuracy is represented in the F1 score.

"""


import itertools


def precision_recall(thresh_results, test_records):

    """
    Imports the results of the frequency-finding and thresholding algorithm, which are the results predicted to be in
    future components. The functions and flows for the components in the testing set are predicted based on these
    results, which are compared with the actual results and used to calculate the accuracy of how well the prediction
    performed.

    +--------+-----+---------------------------------+
    |              |            Predicted?           |
    +--------+-----+----------------+----------------+
    |              | Yes            | No             |
    +--------+-----+----------------+----------------+
    |        | Yes | True Positive  | False Negative |
    |Actual? +-----+----------------+----------------+
    |        | No  | False Positive | True Negative  |
    +--------+-----+----------------+----------------+

    TP = True Positive, FP = False Positive, FN = False Negative, TN = True Negative

    Precision is the ratio of correct predictions to all predictions made by the classifier (TP/(TP + FP)).
    This number is the ratio of predictions that were identified as being in the product that are actually in the product.

    Recall is the ratio of correct predictions to all actual results made by the classifier (TP/(TP + FN)).
    This number is the ratio of the actual results that were correctly predicted.

    Recall is representative of the confidence that no positives have been missed and precision is
    representative of the confidence in the True Positives.

    The F1 score is the harmonic mean of precision and recall

    (2 * precision * recall) / (precision + recall)

    Parameters
    ----------
    thresh_results : dict
    The results of the "find_top_thresh" function. These are the predicted function and flow combinations for each
    component

    test_records : list
    The function and flow combinations for each component in the testing set, organized in a list.

    Returns
    -------
    learned_dict
    Returns a dictionary of what was learned from the results of the data mining automation

    matched
    A dictionary of the functions and flows that were correctly matched for each component

    overmatched
    A dictionary of the functions and flows that were overmatched for each component

    unmatched
    A dictionary of the functions and flows that were unmatched for each component

    recall
    A single number for the recall score for this combination of testing and training sets

    precision
    A single number for the precision score for this combination of testing and training sets

    f1
    A single number for the F1 score for this combination of testing and training sets

    """

    # Make dictionary of actual CFF combinations from test case
    test_actual = {}

    # Remove duplicates from input set of test components
    k = test_records

    k.sort()
    list(k for k,_ in itertools.groupby(k))

    # Create dictionary of format: {component: [function-flow1, function-flow 2, etc.]}
    for e in k:

        # Make dictionary of list of function-flows for each component
        test_actual.setdefault(e[0], []).append(e[1])


    # List for keeping track of which function-flows happen for each component
    keep_flows = []

    # Dictionary for keeping CFF combinations from the learning set
    learned_dict = dict()

    for k,v in thresh_results.items():

        for vs in v:

            # Append list of all of the function-flows for each component
            keep_flows.append(vs[0])

        # Save list of function-flows for each component
        learned_dict[k] = keep_flows

        # Reset list for each component
        keep_flows = []


    # Empty dictionaries for each category
    overmatched = {}
    matched = {}
    unmatched = {}

    not_found = {}

    # Zeroed number for each factor to sum
    overmatched_factor = 0
    unmatched_factor = 0
    matched_factor = 0

    for k, v in test_actual.items():

        # Skip unclassified components
        if k != 'unclassified':

            if k in learned_dict:

                # Make a set for the lists of function-flows for each component
                actual_flows = set(v)
                learned_flows = set(learned_dict[k])

                # Make dictionary for each component based on which category it falls in to

                # If component is in the learning set but not in the test case, it is overmatched
                overmatched[k] = learned_flows.difference(actual_flows)

                # If component is in the test case but not in the learning set, it is unmatched
                unmatched[k] = actual_flows.difference(learned_flows)

                # If component is in both sets, it is matched
                matched[k] = actual_flows.intersection(learned_flows)


                # Keep running sum of how many function-flows fell into each category
                overmatched_factor += len(overmatched[k])
                unmatched_factor += len(unmatched[k])
                matched_factor += len(matched[k])

            else:

                unmatched_factor += len(test_actual[k])


    ## Precision and Recall Stuff

    true_positive = matched_factor
    false_positive = overmatched_factor
    false_negative = unmatched_factor

    recall = true_positive / (true_positive + false_negative)

    precision = true_positive / (true_positive + false_positive)

    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * ((precision * recall)/(precision + recall))

    return learned_dict, matched, overmatched, unmatched, recall, precision, f1




