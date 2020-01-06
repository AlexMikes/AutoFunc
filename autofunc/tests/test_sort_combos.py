from autofunc.counter_pandas import counter_pandas
import os.path
import pandas as pd


def test_1():

    """
    Testing that the highest confidence result for the screw component is couple solid, which is what
    a screw does almost exclusively
    """

    script_dir = os.path.dirname(__file__)
    file_to_test = os.path.join(script_dir, '../assets/consumer_systems.csv')

    test_data = pd.read_csv(file_to_test)

    combos_sorted = counter_pandas(test_data)

    assert combos_sorted['screw'][0][0] == 'couple solid'

