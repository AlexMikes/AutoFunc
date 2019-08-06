from apyori import apriori

"""

Use association rule finding algorithms to find associations between component,
function, and flow in the repository data.

"""


def find_associations(store_data, records, support=0.0003, confidence=0.01, lift=0.1):

    """
        Uses apyori library to find association rules using the apriori algorithm within an itemset

        Parameters
        ----------
        store_data : Pandas data frame
            Data from a .csv file in Pandas data frame format
        records : List
            Same data as store_data but in a list
        support : float
            Threshold for support in apriori algorithm
        confidence : float
            Threshold for confidence in apriori algorithm
        lift : float
            Threshold for lift in apriori algorithm

        Returns
        -------
        conf_results
            Returns a dictionary of function and flow combinations sorted by confidence. The key is the
            component and the value is a list of type: [function-flow, confidence]
        results
            Returns a dictionary of function and flow combinations for each component with the values of
            support, confidence, and lift for each

    """


    # Run apyori to get association rules
    association_rules=apriori(records, min_support=support, min_confidence=confidence, min_lift=lift, min_length=0, max_length=2)

    # Convert to iterable format
    association_results=list(association_rules)


    # Create dictionary in format: {component: (function-flow)}
    results = {}
    conf_results = {}

    ls = []

    # Association Rule head is the component(first column of csv)
    heads = list(store_data['comp'])

    # Loop through list of results from Apyori library
    for e in association_results:

        # For each columm of data
        for j in range(len(e[2])):

            # Convert to string and slice crap off front and back to get just the component
            head = str(e[2][j][0])[12:-3]

            # Some heads are blank for itemsets, ignore those
            if head is not '':

                # Some query results are 'unclassified', ignore those
                if head != 'unclassified':

                    # Only keep association rules where the component is the head
                    if head in heads:
                        support = e[1]

                        confidence = e[2][j][2]

                        lift = e[2][j][3]

                        body = str(e[2][j][1])[12:-3]

                        # Append list of all association rules of interest (function-flow, support, conf, lift)
                        ls.append([head, body, support, confidence, lift])

                        # Dictionary key is component
                        # Append list to values of dictionary for each function-flow along with supp, conf, lift
                        results.setdefault(head, []).append([body, support, confidence, lift])

                        # Store separate dictionary with only confidence results for thresholding later
                        conf_results.setdefault(head, []).append([body, confidence])

    # Sort confidence from largest to smallest
    for k, v in conf_results.items():
        v.sort(key=lambda x: x[1], reverse=True)

    return conf_results, results

