from IceCube.Essential import *
from IceCube.AngularDist import angular_dist_score


def Exam(submission_csv):

    reco_df = pd.read_csv(submission_csv)
    true_df = get_target_angles(BATCHES_TEST)
    LOGGER.info(f"Examing {submission_csv} "
                "with {FILE_TEST_META}, BATCHES = {BATCHES_TEST}")

    score = angular_dist_score(
        true_df["azimuth"], true_df["zenith"],
        reco_df["azimuth"], reco_df["zenith"]
    )

    LOGGER.info(f"Score = {score}")


if __name__ == "__main__":
    Exam("submission.csv")
