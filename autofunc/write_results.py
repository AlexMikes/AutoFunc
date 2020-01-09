'''

Writes a dictionary to a .csv file in a format helpful for visualizing product information

'''

import csv

def write_results_from_dict(learned_dict, outfile, titles = None):
    """
        Writes a dictionary to a .csv file

        Parameters
        ----------
        learned_dict : dict
            The dictionary to write to a file. Can be the result of find_top_thresh
        outfile : str
            The file name and/or path to write the .csv to

        Returns
        -------
        None

        """

   # Write dictionary to csv file
    with open(outfile, 'w') as csv_file:
        writer = csv.writer(csv_file)

        if titles:
            writer.writerow([e for e in titles])
            for key, value in learned_dict.items():
                for vs in value:
                    if len(vs[0]) > 1:
                        writing = [key]
                        for i in range(len(vs)):
                            writing.append(vs[i])
                        writer.writerow(writing)
                    else:
                        writer.writerow([key,vs])


        else:
            for key, value in learned_dict.items():
                for vs in value:
                    if len(vs[0]) > 1:
                        writing = [key]
                        for i in range(len(vs)):
                            writing.append(vs[i])
                        writer.writerow(writing)
                    else:
                        writer.writerow([key,vs])

