"""
This counts instances of function-flow combinations per component to find the frequency of each combination by
dividing it by all of the combinations for each component (# of occurrences / total occurrences).

The output includes some of the intermediate data structures that can be used for finding other numbers besides
frequency.

"""

def counter_pandas_with_counts(dataframe):

    """
        Counts instances and sorts them by frequency

        Parameters
        ----------
        dataframe : Pandas dataframe
            A Pandas dataframe with the product information

        Returns
        -------
        comb_sort
            Returns a dictionary of function and flow combinations sorted by frequency. The key is the
            component and the value is a list of the structure: [function-flow, frequency]

        counts
            Returns a dictionary of the total number of function and flow combinations for each component

        combos
            Returns a dictionary of the number of individual instances of function and flow combinations per component

    """

    # Combinations of components, functions, and/or flows are stored in a dictionary with the first column
    # as the key and the second column as the value

    combos = {}

    # Instances of each item in the columns are counted for later analysis
    counts = {}

    for row in dataframe.itertuples():


        # By convention, the first column is the component and the second column is the function and/or flow
        comp = row.comp
        func = row.func

        # Create a dictionary with a count of instances of each component
        if comp not in counts:
            counts[comp] = 1
        else:
            counts[comp] += 1

        # Create a dictionary that tracks the number of times a component has a function and/or flow
        if comp not in combos:
            combos[comp] = {}

            combos[comp][func] = 1

        else:
            if func not in combos[comp]:
                combos[comp][func] = 1
            else:
                combos[comp][func] += 1

    # (1) Convert the dictionary of a dictionary to a dictionary of lists for sorting then (2) divide the functions
    # and/or flows for each component by the total number of component instances to get the percentage
    # of each combination and (3) sort the dictionary by the percentages of each combination.

    # (1) Convert
    comb_sort = {}
    for cs, fs in combos.items():
        for k, v in combos[cs].items():
            # (2) Divide
            # comb_sort.setdefault(cs, []).append([k, v / counts[cs], '{0}/{1}'.format(v,len(dataframe))])#  v/len(dataframe)])

            # Two columns for support because excel can't handle not dividing them
            comb_sort.setdefault(cs, []).append([k, v / counts[cs], v, len(dataframe)])

    # (3) Sort
    for k, v in comb_sort.items():
        v.sort(key=lambda x: x[1], reverse=True)

    return comb_sort, counts, combos




