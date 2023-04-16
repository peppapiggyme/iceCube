from IceCube.Essential import *
from IceCube.Model import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import threading
import pdb


def BoostedDecisionTree(X, y, tree_args, boosting_args):
    dt = DecisionTreeClassifier(**tree_args)
    clf = AdaBoostClassifier(base_estimator=dt, **boosting_args)

    # Train classifier on training set
    clf.fit(X, y)

    # Test classifier on testing set
    score = clf.decision_function(X)
    y_hat = clf.predict(X)

    return y_hat, score, clf


def Train(X, y, tree_args, boosting_args, tag):
    # train the model
    LOGGER.info(f"Starting training ... {tag}")
    y_hat, _, clf = BoostedDecisionTree(X, y, tree_args, boosting_args)

    # Evaluate accuracy
    accuracy = accuracy_score(y, y_hat)
    LOGGER.info(f"train accuracy of {tag} -> {accuracy*100:.2f}%")

    # save the model
    LOGGER.info(f"Finished training, saving model ... {tag}")
    pickle.dump(clf, open(os.path.join(
        MODEL_PATH, f"EventCat_clf.{tag}.sklearn"), "wb"))


if __name__ == "__main__":
    # batches extracted by GNN_test.py
    batches = list(range(11, 16))
    # batches = [1]

    LOGGER.info(f"{len(batches)} files for BDT training")
    threads = list()

    # ground truth
    true_df = get_target_angles(batches)
    true_df = angles2vector(true_df)
    print(true_df.head(5))

    # reconstructed directions
    reco_df = get_reco_angles(batches)
    reco_df[np.isnan(reco_df)] = 0
    print(reco_df.head(5))

    n = true_df[["nx", "ny", "nz"]].to_numpy()
    n_hat = reco_df[["x", "y", "z"]].to_numpy()

    error, az_error, ze_error = angle_errors(n, n_hat)
    print(
        f"error, az_error, ze_error = {error.mean()}, {az_error.mean()}, {ze_error.mean()}")

    idx = reco_df.kappa.values < 0.5

    # reco_df inputs
    X = reco_df[col_eventCat].to_numpy()

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth": 3,
        "random_state": SEED,
    }

    boosting_args = {
        "n_estimators": 10,
        "learning_rate": 0.8,
        "random_state": SEED,
    }

    # inputs
    LOGGER.info(f"input shape = {X.shape}")

    threads.append(
        threading.Thread(target=Train, args=(
            X, idx, tree_args, boosting_args, "Tree.10"))
    )

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth": 3,
        "random_state": SEED,
    }

    boosting_args = {
        "n_estimators": 8,
        "learning_rate": 0.8,
        "random_state": SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(
            X, idx, tree_args, boosting_args, "Tree.8"))
    )

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth": 3,
        "random_state": SEED,
    }

    boosting_args = {
        "n_estimators": 6,
        "learning_rate": 0.8,
        "random_state": SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(
            X, idx, tree_args, boosting_args, "Tree.6"))
    )

    # starting training
    _ = [t.start() for t in threads]
    _ = [t.join() for t in threads]

    LOGGER.info("All threads are finished")
