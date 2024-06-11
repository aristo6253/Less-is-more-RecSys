import numpy as np
import h5py
from interactions import Interactions
import pandas as pd


# The code was taken from: https://maciejkula.github.io/spotlight/_modules/spotlight/
def getData():
    # with h5py.File("data/goodbooks-10k.hdf5") as data:
    #     return (
    #         data["ratings"][:100000, 0].astype(np.int32),
    #         data["ratings"][:100000, 1].astype(np.int32),
    #         data["ratings"][:100000, 2].astype(np.float32),
    #         # TODO: replace for all instances 
    #         #data["ratings"][:, 0].astype(np.int32),
    #         #data["ratings"][:, 1].astype(np.int32),
    #         #data["ratings"][:, 2].astype(np.float32),
    #         np.arange(len(data["ratings"]), dtype=np.int32),
    #     )
    dfn = pd.read_csv('sorted_ratings.csv')
    dfn = dfn[dfn['user_id'] <= 5000]
    user_ids = dfn['user_id'].values.astype(np.int32)
    book_ids = dfn['book_id'].values.astype(np.int32)
    ratings = dfn['rating'].values.astype(np.float32)
    # t = np.arange(len(dfn["rating"]), dtype=np.int32)
    return (user_ids, book_ids, ratings)



def get_goodbooks_dataset():
    data = getData()
    # binary data
    interactions = Interactions(user_ids=data[0], item_ids=data[1])
    # interactions = Interactions(user_ids=data[0], item_ids=data[1], ratings=data[2] / 5)
    # ratio data 1 - 5
    # interactions = Interactions(data[0], data[1], data[2])

    trainData, testData = random_train_test_split(interactions, 0.2)
    data = [trainData.tocoo(), testData.tocoo()]
    return data


def _index_or_none(array, shuffle_index):

    if array is None:
        return None
    else:
        return array[shuffle_index]


def shuffle_interactions(interactions, random_state=None):
    """
    Shuffle interactions.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    interactions: :class:`spotlight.interactions.Interactions`
        The shuffled interactions.
    """

    # TODO: fix random_state
    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(interactions.user_ids))
    random_state.shuffle(shuffle_indices)

    return Interactions(
        interactions.user_ids[shuffle_indices],
        interactions.item_ids[shuffle_indices],
        ratings=_index_or_none(interactions.ratings, shuffle_indices),
        timestamps=_index_or_none(interactions.timestamps, shuffle_indices),
        weights=_index_or_none(interactions.weights, shuffle_indices),
        num_users=interactions.num_users,
        num_items=interactions.num_items,
    )


def random_train_test_split(interactions, test_percentage=0.2, random_state=None):
    """
    Randomly split interactions between training and testing.

    Parameters
    ----------

    interactions: :class:`spotlight.interactions.Interactions`
        The interactions to shuffle.
    test_percentage: float, optional
        The fraction of interactions to place in the test set.
    random_state: np.random.RandomState, optional
        The random state used for the shuffle.

    Returns
    -------

    (train, test): (:class:`spotlight.interactions.Interactions`,
                    :class:`spotlight.interactions.Interactions`)
         A tuple of (train data, test data)
    """

    interactions = shuffle_interactions(interactions, random_state=random_state)

    cutoff = int((1.0 - test_percentage) * len(interactions))

    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)

    train = Interactions(
        interactions.user_ids[train_idx],
        interactions.item_ids[train_idx],
        ratings=_index_or_none(interactions.ratings, train_idx),
        timestamps=_index_or_none(interactions.timestamps, train_idx),
        weights=_index_or_none(interactions.weights, train_idx),
        num_users=interactions.num_users,
        num_items=interactions.num_items,
    )
    test = Interactions(
        interactions.user_ids[test_idx],
        interactions.item_ids[test_idx],
        ratings=_index_or_none(interactions.ratings, test_idx),
        timestamps=_index_or_none(interactions.timestamps, test_idx),
        weights=_index_or_none(interactions.weights, test_idx),
        num_users=interactions.num_users,
        num_items=interactions.num_items,
    )

    return train, test


get_goodbooks_dataset()
