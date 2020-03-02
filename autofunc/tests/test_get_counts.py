from autofunc.counter_pandas_with_counts import counter_pandas_with_counts
import os.path
from autofunc.make_df import make_df

def test1():

""" Testing the counter function. When using the consumer_systems dataset, housing is the most common component """

    script_dir = os.path.dirname(__file__)
    file1 = os.path.join(script_dir, '../assets/consumer_systems.csv')

    # Pandas
    df = make_df(file1)

    comb_sort, counts, combos = counter_pandas_with_counts(df)

    max_component_name = max(counts, key=counts.get)

    assert max_component_name == 'housing'