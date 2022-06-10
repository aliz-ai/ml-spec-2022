import argparse
import datetime
import warnings

import hypertune

from modelling.models import save_info, save_model, train_eval_k_fold
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

BUCKET_NAME = "aliz-ml-spec-2022"
JOB_DIR = "demo-2/modelling_serializations"


if __name__ == "__main__":
    hparams = {
        "n_jobs": -1,
    }

    regressor = LinearRegression(**hparams)
    model, cv_rmse, test_eval_metrics = train_eval_k_fold(regressor)

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
