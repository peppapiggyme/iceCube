from IceCube.Essential import *
from IceCube.Model import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle, threading
import pdb


def BoostedDecisionTree(X, y, tree_args, boosting_args):
    dt = DecisionTreeClassifier(**tree_args)
    clf = AdaBoostClassifier(base_estimator=dt, **boosting_args)

    # Train classifier on training set
    clf.fit(X, y)

    # Test classifier on testing set
    score = clf.decision_function(X)
    y_hat = clf.predict(X)
    
    accuracy = accuracy_score(y, y_hat)
    LOGGER.info(f"Train accuracy: {accuracy * 100:.2f}%")

    return y_hat, score, clf


def Train(X, y, error, errorx, tree_args, boosting_args, tag):
    # train the model
    LOGGER.info(f"Starting training ... {tag}")
    y_hat, _, clf = BoostedDecisionTree(X, y, tree_args, boosting_args)
    # save the model
    LOGGER.info(f"Finished training, saving model ... {tag}")
    pickle.dump(clf, open(os.path.join(MODEL_PATH, "BDT_clf.", tag, ".sklearn"), "wb"))

    # Evaluate accuracy
    error[y_hat] = errorx[y_hat]
    LOGGER.info(f"error of {tag}-> {error.mean()}")


if __name__ == "__main__":
    # batches extracted by GNN_test.py
    batches = list(range(1, 21))
    threads = list()

    # region prepare

    # ground truth
    true_df = get_target_angles(batches)
    true_df = angles2vector(true_df)
    print(true_df.head(5))
    n = true_df[["nx","ny","nz"]].to_numpy()

    # reconstructed directions
    reco_df = get_reco_angles(batches)
    print(reco_df.head(5))
    n_hat = reco_df[["x", "y", "z"]].to_numpy()

    e = reco_df[["ex", "ey", "ez"]].to_numpy()
    xe = np.sum(n_hat * e, axis=1)
    print(xe.shape)
    proj = n_hat - xe[:, np.newaxis] * e
    proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)

    error, az_error, ze_error = angle_errors(n, n_hat)
    print(f"error, az_error, ze_error = {error.mean()}, {az_error.mean()}, {ze_error.mean()}")

    errorx, az_errorx, ze_errorx = angle_errors(n, proj)
    print(f"error, az_error, ze_error = {errorx.mean()}, {az_errorx.mean()}, {ze_errorx.mean()}")

    idx = error > errorx

    # reco_df inputs
    reco = reco_df[["fit_error", "sumc", "hits", "zenith", "ez", "dt", "unique_x", "unique_z"]].to_numpy()
    reco[:, 0] = np.log10(reco[:, 0] + 1e-8)
    reco[:, 1] = np.log10(reco[:, 1] + 1e-8)

    # inputs
    X = np.concatenate([reco, np.abs(xe[:, np.newaxis])], axis=1)
    LOGGER.info(f"input shape = {X.shape}")

    # endregion

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 2, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 800, 
        "learning_rate" : 0.8, 
        "random_state" : SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "Baseline"))
    )

    # -------------------------------------------------------------------------
    # Deep
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 3, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 800, 
        "learning_rate" : 0.8, 
        "random_state" : SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "Deep"))
    )

    # -------------------------------------------------------------------------
    # LowLR
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 2, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 800, 
        "learning_rate" : 0.6, 
        "random_state" : SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "LowLR"))
    )

    # -------------------------------------------------------------------------
    # HighLR
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 2, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 800, 
        "learning_rate" : 1.0, 
        "random_state" : SEED,
    }

    # -------------------------------------------------------------------------
    # DeepLowLR
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 3, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 800, 
        "learning_rate" : 0.6, 
        "random_state" : SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "DeepLowLR"))
    )

    # -------------------------------------------------------------------------
    # DeepHighLR
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 3, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 800, 
        "learning_rate" : 1.0, 
        "random_state" : SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "DeepHighLR"))
    )

    # -------------------------------------------------------------------------
    # DeepLess
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 3, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 400, 
        "learning_rate" : 0.8, 
        "random_state" : SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "DeepLess"))
    )

    # -------------------------------------------------------------------------
    # DeepMore
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 3, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 1200, 
        "learning_rate" : 0.8, 
        "random_state" : SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "DeepMore"))
    )

    # starting training
    _ = [t.start() for t in threads]
    _ = [t.join()  for t in threads]

    LOGGER.info("All threads are finished")
