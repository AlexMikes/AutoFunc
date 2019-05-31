from autofunc.simple_counter import count_stuff, find_top_thresh
import os.path


def test_count_stuff():

    """
    Tests that the top function-flow combination for the component "screw" is "couple solid"
    """

    script_dir = os.path.dirname(__file__)
    file1 = os.path.join(script_dir, '../assets/bladeCombined.csv')

    comb_sort = count_stuff(file1)

    assert comb_sort['screw'][0][0] == 'couple solid'


def test_find_top_thresh():

    """
    Tests that the top 70% of function-flow combinations for the component "screw" only has one result
    """

    script_dir = os.path.dirname(__file__)
    file2 = os.path.join(script_dir, '../assets/bladeCombined.csv')

    comb_sort = count_stuff(file2)

    threshold = 0.7
    thresh_results = find_top_thresh(comb_sort, threshold)

    assert len(thresh_results['screw']) == 1


