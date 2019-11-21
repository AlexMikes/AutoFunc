import pickle
import scipy.stats as stats



with open("bd.txt", "rb") as fp:  # Unpickling
    bd = pickle.load(fp)

with open("same.txt", "rb") as fp:  # Unpickling
    same = pickle.load(fp)

with open("consumer.txt", "rb") as fp:  # Unpickling
    consumer = pickle.load(fp)


# tillamook = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735,0.0659, 0.0923, 0.0836]
# newport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725]
# petersburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
# magadan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764,0.0689]
# tvarminne = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]
