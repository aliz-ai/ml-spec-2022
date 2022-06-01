import argparse
import datetime

import hypertune

from trainer.model import save_info, save_model, train_and_evaluate

BUCKET_NAME = "aliz-ml-spec-2022"
JOB_DIR = "demo-2/modelling_serializations"


if __name__ == "__main__":

    # provide training job parameters as arguments for hyperparameter tuning job
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_depth",
        help="Maximum depth of the individual estimators",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--learning_rate", help="learning rate", type=float, default=0.2
    )
    parser.add_argument(
        "--n_estimators", help="Number of boosting iterations", type=int, default=500
    )
    parser.add_argument(
        "--gamma",
        help="Minimum loss reduction required to make a further partition on a leaf node of the tree",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--subsample",
        help="Subsample ratio of the training instance",
        default=1,
        type=float,
    )
    args, _ = parser.parse_known_args()

    hparams = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "gamma": args.gamma,
        "subsample": args.subsample,
        "tree_method": "hist",
        "n_jobs": -1,
    }

    model, cv_rmse, test_eval_metrics = train_and_evaluate(hparams)

    # serialize
    name = []
    for k, v in sorted(hparams.items()):
        name.append("%s=%s" % (k, v))
    name = "__".join(name)
    directory = "{}/{}".format(
        JOB_DIR, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    save_model(model, name, bucket_name=BUCKET_NAME, directory=directory)
    save_info(
        test_eval_metrics,
        "[test_eval_metrics] " + name,
        bucket_name=BUCKET_NAME,
        directory=directory,
    )

    # hyperparameter tuning
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="rmse", metric_value=cv_rmse, global_step=1
    )
