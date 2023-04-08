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
    
    return y_hat, score, clf


def Train(X, y, error, errorx, tree_args, boosting_args, tag):
    # train the model
    LOGGER.info(f"Starting training ... {tag}")
    y_hat, _, clf = BoostedDecisionTree(X, y, tree_args, boosting_args)

    # Evaluate accuracy
    accuracy = accuracy_score(y, y_hat)
    LOGGER.info(f"train accuracy of {tag} -> {accuracy*100:.2f}%")
    error[y_hat] = errorx[y_hat]
    LOGGER.info(f"error of {tag} -> {error.mean()}")

    # save the model
    LOGGER.info(f"Finished training, saving model ... {tag}")
    pickle.dump(clf, open(os.path.join(MODEL_PATH, f"BDT_clf.{tag}.sklearn"), "wb"))


if __name__ == "__main__":
    # batches extracted by GNN_test.py
    batches = list(range(1, 11))
    # batches = [1]
    threads = list()

    # ground truth
    true_df = get_target_angles(batches)
    true_df = angles2vector(true_df)
    print(true_df.head(5))
    n = true_df[["nx","ny","nz"]].to_numpy()

    # reconstructed directions
    reco_df = get_reco_angles(batches)
    reco_df[np.isnan(reco_df)] = 0
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
    reco_df["fit_error"] = np.log10(reco_df["fit_error"] / reco_df["hits"] + 1e-6)
    reco_df["sumw"] = np.log10(reco_df["sumw"] + 1e-3)
    reco_df["sumc"] = np.log10(reco_df["sumc"] + 1e-3)
    reco_df["sumt"] = np.log10(reco_df["sumt"] + 1e-3)
    reco_df["dt"] = np.log10(reco_df["dt"] + 1e-3)
    reco_df["std_t"] = np.tanh(reco_df["std_t"])
    reco_df["std_z"] = np.tanh(reco_df["std_z"])
    reco = reco_df[["fit_error", "sumw", "sumc", "sumt", "dt", "std_t", "std_z", "unique_x", "zenith", "ez"]].to_numpy()
    xe = np.arccos(xe)

    # inputs
    X = np.concatenate([reco, xe[:, np.newaxis]], axis=1)
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

