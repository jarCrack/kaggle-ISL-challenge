import warnings

import numpy as np
import pandas as pd
import prefixes
from sklearn.cross_validation import train_test_split
from tqdm import tqdm
from collections import namedtuple, Counter

AR_METHOD = 1
PREFIX_METHOD = 2
MODE_METHOD = 3

use_frequency_prediction = False


def warn(*args, **kwargs):
    pass


warnings.warn = warn

import sklearn.linear_model
from sklearn.preprocessing import PolynomialFeatures

ResultLog = namedtuple('ResultLog', 'sequence_id original_value predicted_value prediction_method big_number')


# train model for each series

def linear_regression(X, y):
    clf = sklearn.linear_model.LinearRegression(fit_intercept=True)
    clf.fit(X, y)
    return clf


def prepare_instances(sequence, window_len=3):
    train_len = len(sequence) - (window_len)
    dat = np.zeros([train_len, window_len + 1], dtype=object)
    for i in range(train_len):
        dat[i] = sequence[i:i + window_len + 1]
    # print(dat)

    X = pd.DataFrame(dat, columns=list(map(str, range(window_len))) + ["y"])
    y = X.pop("y")
    return X, y


def fib(n):
    a, b = 1, 1
    for i in range(n - 1):
        a, b = b, a + b
    return a


def store_results(results):
    res = pd.DataFrame([res.predicted_value for res in results], index=[res.sequence_id for res in results], columns=['Last'])
    res.index.name = 'Id'
    res.to_csv("result.csv")


def polynomial_features(X, order=2):
    poly = PolynomialFeatures(order, include_bias=False)
    X = poly.fit_transform(X.as_matrix())
    return pd.DataFrame(X)


def rounded_mode(seq):
    return int(max(set(seq), key=seq.count))


trial_series = [0] + list(map(fib, range(1, 30)))
print(trial_series)
vals = []
orig_vals = []


def create_frequency_table(sequences, test_run):
    number_store = {}
    for idx, el in tqdm(sequences.iterrows(), total=len(sequences)):
        if test_run:
            seq = list(map(float, el.Sequence.split(',')))[:-1]
        else:
            seq = list(map(float, el.Sequence.split(',')))
        if (len(seq) > 2):
            for i, val in enumerate(seq[:-1]):
                if val in number_store:
                    if seq[i + 1] in number_store[val]:
                        number_store[val][seq[i + 1]] += 1
                    else:
                        number_store[val][seq[i + 1]] = 1
                else:
                    number_store[val] = {}
                    number_store[val][seq[i + 1]] = 1

    return number_store


if __name__ == "__main__":
    test_run = True
    test_data = pd.read_csv("../data/test.csv")
    train_data = pd.read_csv("../data/train.csv")

    # print(train_data.iloc[0][1])

    if use_frequency_prediction:
        number_store = create_frequency_table(test_data, test_run)

    if test_run:
        train_data = train_data.head(10000)
        test_data = test_data.head(1000)

    combi = train_data


    # combi = pd.concat([train_data, test_data])


    def evaluate(idx, el):
        found = False
        seq = list(map(float, el.Sequence.split(',')))
        # seq = trial_series

        orig_val = seq[-1]
        if test_run:
            seq = seq[:-1]

        max_AR_order = len(seq) - 1

        AR_used = False

        prediction_method = PREFIX_METHOD
        val = prep.findNextAndDerive(list(map(int, seq)))

        if len(seq) > 3:
            big_num = seq[-2] > 1e17
        else:
            big_num = False
        if val is None:
            for AR_order in range(2, max_AR_order):

                # To avoid overfitting there must be minmum as many instances as features
                # An AR[n] process yields k/2*(3+k) polynomial features for poly order up to two.

                X_orig, y_orig = prepare_instances(seq, window_len=AR_order)
                X_orig=polynomial_features(X_orig,order=1)


                X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, random_state=42,
                                                                    test_size=0.16)

                clf = linear_regression(X_train, y_train)
                y_pred = list(map(round, clf.predict(X_test)))

                if list(y_test) == y_pred:
                    y_pred = clf.predict(polynomial_features(pd.Series(seq[-AR_order:]),order=1))
                    found = True
                    break

            if found:
                prediction_method = AR_METHOD
                val = int(round(y_pred[0]))
            else:
                if len(seq) > 0:
                    """
                    if seq[-1] in number_store:
                        values = list(number_store[seq[-1]].values())
                        keys = list(number_store[seq[-1]].keys())
                        numbers = max(values)
                        if (float(numbers) / sum(values) > 0.3):
                            val = keys[values.index(numbers)]
                        else:
                            val = rounded_mode(seq)
                    else:
                        val = rounded_mode(seq)
                    """
                    # val=prep.findNextAndDerive(list(map(int,seq)))
                    val = rounded_mode(seq)
                    prediction_method = MODE_METHOD
                else:
                    val = 0

        guessed_right = orig_val == val
        return ResultLog(original_value=orig_val, predicted_value=val, sequence_id=el.Id,
                         prediction_method=prediction_method, big_number=big_num)


    from joblib import Parallel, delayed
    import multiprocessing

    # Prepare datrie for prefix search
    seqs = {el.Id: list(map(int, el.Sequence.split(','))) for idx, el in
            tqdm(combi.iterrows(), total=len(combi))}

    prep = prefixes.Preparator()
    prep.prepare_Trie(seqs)

    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(evaluate)(idx, el) for idx, el in tqdm(test_data.iterrows(), total=len(test_data)))

    store_results(results)

    num_AR_detected = sum(res.prediction_method == AR_METHOD for res in results)
    num_correct_ARs = sum(
        res.prediction_method == AR_METHOD and res.predicted_value == res.original_value for res in results)

    num_prefix_detected=sum(res.prediction_method == PREFIX_METHOD for res in results)
    num_correct_prefix=sum(
        res.prediction_method == PREFIX_METHOD and res.predicted_value == res.original_value for res in results)

    num_correct_predictions=sum(res.predicted_value==res.original_value for res in results)
    big_numbs = sum(res.big_number for res in results)


    print(num_correct_prefix/num_prefix_detected)
    print(num_correct_ARs/num_AR_detected)
    print(big_numbs / len(test_data))
    print(num_correct_predictions/(len(test_data)))
