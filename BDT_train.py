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
    batches = list(range(1, 81))
    # batches = list(range(1, 6))
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

    n = true_df[["nx","ny","nz"]].to_numpy()
    n_hat = reco_df[["x", "y", "z"]].to_numpy()

    e = reco_df[["ex", "ey", "ez"]].to_numpy()
    xe = np.sum(n_hat * e, axis=1, keepdims=True)
    print(xe.shape)
    proj = n_hat - xe * e
    proj /= (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)

    error, az_error, ze_error = angle_errors(n, n_hat)
    print(f"error, az_error, ze_error = {error.mean()}, {az_error.mean()}, {ze_error.mean()}")

    errorx, az_errorx, ze_errorx = angle_errors(n, proj)
    print(f"error, az_error, ze_error = {errorx.mean()}, {az_errorx.mean()}, {ze_errorx.mean()}")

    idx = error > errorx

    # reco_df inputs
    reco_df["error"] = np.log10(reco_df["error"] + 1e-6)
    reco_df["sumq"] = np.log10(reco_df["sumq"] + 1e-3)
    reco_df["dt_15"] = np.log10(reco_df["dt_15"] + 1e-3)
    reco_df["dt_50"] = np.log10(reco_df["dt_50"] + 1e-3)
    reco_df["dt_85"] = np.log10(reco_df["dt_85"] + 1e-3)
    reco_df["kappa"] = np.log10(reco_df["kappa"] + 1e-3)
    columns = ["kappa", "zenith", "error", "sumq", "qz", "dt_50", "dt_85", "ez"]
    reco = reco_df[columns].to_numpy()
    xe = np.arccos(xe)

    # trajectory display
    col_xyzt = [
        "x0", "y0", "z0", "t0",
        "x1", "y1", "z1", "t1",
        "x2", "y2", "z2", "t2",
        "x3", "y3", "z3", "t3", ]
    traj = reco_df[col_xyzt].values
    traj = traj.reshape(-1, 4, 4)

    v1 = 1e3 * (traj[:, 1, :3] - traj[:, 0, :3]) / (traj[:, 1, 3] - traj[:, 0, 3] + 1)[:, np.newaxis]
    v2 = 1e3 * (traj[:, 2, :3] - traj[:, 1, :3]) / (traj[:, 2, 3] - traj[:, 1, 3] + 1)[:, np.newaxis]
    v3 = 1e3 * (traj[:, 3, :3] - traj[:, 2, :3]) / (traj[:, 3, 3] - traj[:, 2, 3] + 1)[:, np.newaxis]

    v1scale = np.linalg.norm(v1, axis=1, keepdims=True) + 1e-1
    v2scale = np.linalg.norm(v2, axis=1, keepdims=True) + 1e-1
    v3scale = np.linalg.norm(v3, axis=1, keepdims=True) + 1e-1

    ev1 = np.sum(-v1 * e / v1scale, axis=1, keepdims=True)
    ev2 = np.sum(-v2 * e / v2scale, axis=1, keepdims=True)
    ev3 = np.sum(-v3 * e / v3scale, axis=1, keepdims=True)

    ev1 = np.arccos(ev1)
    ev2 = np.arccos(ev2)
    ev3 = np.arccos(ev3)

    vv12 = np.sum(v1 * v2 / v1scale / v2scale, axis=1, keepdims=True)
    vv23 = np.sum(v2 * v3 / v2scale / v3scale, axis=1, keepdims=True)
    vv31 = np.sum(v3 * v1 / v3scale / v1scale, axis=1, keepdims=True)

    vavg = np.log10(np.mean((v1scale, v2scale, v3scale), axis=0))
    evvv = np.mean((ev1, ev2, ev3), axis=0)
    vvvv = np.mean((vv12, vv23, vv31), axis=0)

    pos = np.mean(traj[:, :, :3], axis=1)
    xyzq = reco_df[["qx", "qy", "qz"]].to_numpy()
    distq = pos - xyzq
    distq = np.linalg.norm(distq, axis=1, keepdims=True) + 1e-3

    # input
    X = np.concatenate([reco, xe, ev1, ev2, ev3, vavg, evvv, vvvv, distq], axis=1)
    LOGGER.info(f"input shape = {X.shape}")

    # # -------------------------------------------------------------------------
    # # Test
    # # -------------------------------------------------------------------------
    # tree_args = {
    #     "max_depth" : 3, 
    #     "random_state" : SEED, 
    # }

    # boosting_args = {
    #     "n_estimators" : 400, 
    #     "learning_rate" : 0.8, 
    #     "random_state" : SEED,
    # }

    # # inputs
    # X = np.concatenate([reco, xe, ev1, ev2, ev3, vavg, evvv, vvvv, distq], axis=1)
    # LOGGER.info(f"input shape = {X.shape}")

    # threads.append(
    #     threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "Test"))
    # )

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
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "Baseline.0414"))
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
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "Deep.0414"))
    )

    # -------------------------------------------------------------------------
    # BaseMore
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 2, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 1200, 
        "learning_rate" : 0.8, 
        "random_state" : SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "BaseMore.0414"))
    )

    # -------------------------------------------------------------------------
    # BaseMoreMore
    # -------------------------------------------------------------------------
    tree_args = {
        "max_depth" : 2, 
        "random_state" : SEED, 
    }

    boosting_args = {
        "n_estimators" : 1600, 
        "learning_rate" : 0.8, 
        "random_state" : SEED,
    }

    threads.append(
        threading.Thread(target=Train, args=(X, idx, error, errorx, tree_args, boosting_args, "BaseMoreMore.0414"))
    )

    # starting training
    _ = [t.start() for t in threads]
    _ = [t.join()  for t in threads]
    
    LOGGER.info("All threads are finished")

