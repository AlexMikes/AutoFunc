import itertools


def precision_recall(thresh_results, test_records):

    """
    Compares the results learned from data mining with a verification case for which the actual results are known
    and outputs a "Match Factor" based on how well the automation matched the actual results as if they weren't known.

    Imports the results of the thresholding as well as the data from the verification case.

    The match factor is a ratio of correct to incorrect learning. Correct means a function and flow is learned for a
    component and that function and flow exist in the verification case for that component.

    Incorrect means one of two things. It could be overmatched, in which a function and flow is learned for a component
    that was not in the verification case. Unmatched means a function and flow exists in the verification case for a
    component that was not learned from the data mining.

    The match factor is then the correct matches divided by the sum of the overmatched and unmatched numbers.

    The outputs are the match factor as a number and dictionaries of each case: matched, overmatched, and unmatched.

    Parameters
    ----------
    thresh_results : dict
    The results of the "find_top_thresh" function
    test_records : dict
    The results of the "get_data" function for the verification test case

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
    match_factor
    A float of the match factor

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


    # Find overall match factor
    # match_factor = matched_factor / (unmatched_factor + overmatched_factor)





    ## Precision and Recall Stuff

    true_positive = matched_factor
    false_positive = overmatched_factor
    false_negative = unmatched_factor

    recall = true_positive / (true_positive + false_negative)

    precision = true_positive / (true_positive + false_positive)

    f1 = 2 * ((precision * recall)/(precision + recall))

    return learned_dict, matched, overmatched, unmatched, recall, precision, f1 #, match_factor




    # return learned_dict, matched, overmatched, unmatched, match_factor