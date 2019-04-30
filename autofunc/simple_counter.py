
class SimpleCounter:

    """ Counts instances of items in .csv file columns """

    def __init__(self, filename):

        self.filename = filename

    def count_stuff(self):

        # Combinations of components, functions, and/or flows are stored in a dictionary with the first column
        # as the key and the second column as the value

        combos = {}

        # Instances of each item in the columns are counted for later analysis
        counts = {}

        with open(self.filename, encoding='utf-8-sig') as input_file:
            for row in csv.reader(input_file, delimiter=','):

                # By convention, the first column is the component and the second column is the function and/or flow
                comp = row[0]
                func = row[1]

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
                comb_sort.setdefault(cs, []).append([k, v / counts[cs]])

        # (3) Sort
        for k, v in comb_sort.items():
            v.sort(key=lambda x: x[1], reverse=True)

        return comb_sort


# if __name__ == "__main__":