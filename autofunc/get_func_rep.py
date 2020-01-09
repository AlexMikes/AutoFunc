"""

Find and return a functional representation of a product based on its constituent components and the results previously
found from data mining

"""

import csv

def get_func_rep(thresh_results, input_comps, conf_values = True):

    """
    Find the functional representation of a set of components based on the results of data mining

    Parameters
    ----------
    thresh_results : dict
       The return dictionary from the "get_top_results" function
    input_comps : string
       The filename of a .csv file containing the components of a product
    conf_values : bool
        A boolean of whether or not to return the results with the frequency values, default is True

    Returns
    -------
    learned_dict
       Returns a dictionary of function and flow combinations sorted by confidence for each component in the input_case.
       The key is the component and the value is a list of function-flow combinations.
    unmatched
        Returns a list of components that were in the set of input components but not found in the data mining results.

    """

    # Instances of each component are counted
    counts = {}

    # keep a list of components that were not learned in data mining
    unmatched = []

    with open(input_comps, encoding='utf-8-sig') as input_file:
        for row in csv.reader(input_file, delimiter=','):

            comp = row[0]
            # Create a dictionary with each component
            if comp not in counts:
                counts[comp] = 1
            else:
                counts[comp] += 1



    # Method of returning results with frequency values
    if conf_values is True:

        res_with_conf = {}

        # Inherit the values of thresh_results for each of the same component in input components
        for k, v in counts.items():

            if k in thresh_results:

                res_with_conf[k] = thresh_results[k]

            else:

                if k not in unmatched:
                    unmatched.append(k)

        return res_with_conf, unmatched

    else:

    # Method of returning results without frequency values

        # List for keeping track of which function-flows happen for each component
        keep_flows = []

        # Dictionary for keeping CFF combinations from the learning set
        learned_dict = {}

        for k, v in counts.items():

            if k in thresh_results:

                for vs in thresh_results[k]:

                    # Append list of all of the function-flows for each component
                    keep_flows.append(vs[0])

                # Save list of function-flows for each component
                learned_dict[k] = keep_flows

                # Reset list for each component
                keep_flows = []

            else:

                if k not in unmatched:
                    unmatched.append(k)

        return learned_dict, unmatched





