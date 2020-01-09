"""

Takes the results of the frequency-finding and sorting algorithm and cuts off the sum of frequencies at a threshold

"""


def get_top_results(conf_results, threshold=0.7):

    """
    Use top XX% of results as a threshold for the functions and flows that are associated with each component.

    Parameters
    ----------
    conf_results : dict
       The return dictionary from the "find_associations" function
    threshold : float
       The threshold used for finding the top percentage of confidences

    Returns
    -------
    return_dict
       Returns a dictionary of function and flow combinations sorted by confidence that sum up to the threshold.
       The key is the component and the value is a list of type: [function-flow, confidence]

    """

    # Empty dictionary to collect top percentage of threshold
    thresh_results = {}

    # Counter to keep track of sum of confidences through the top results
    so_far = 0

    # Counter to keep track of how many function-flows were used to reach sum threshold
    i = 0

    # Sum confidence values for each CFF
    for k,v in conf_results.items():

        while so_far <= threshold:

            # Each value is a list of list, iterate through the outer list
            for vs in v:

                if so_far <= threshold:

                    # Sum confidence values
                    so_far += vs[1]

                    # Keep track of list index for appending resulting CFF combinations
                    i += 1

        # For the results dictionary, append a list of all resulting function-flows that fell within the threshold
        thresh_results.setdefault(k, []).append(v[0:i])

        # Reset index counter and confidence sum
        i = 0
        so_far = 0

    return_dict = {}

    for k,v in thresh_results.items():

        return_dict[k] = v[0]

    return return_dict