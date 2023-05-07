import argparse
import logging
from typing import Callable
from typing import Iterable
from typing import List
from typing import Tuple
from typing import Union

import numpy
import pandas
import scipy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import apache_beam as beam
from apache_beam.ml.inference.base import KeyedModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import RunInference
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.runners.runner import PipelineResult

from lightgbm_inference import LightGBMModelHandlerNumpy
from lightgbm_inference import LightGBMModelHandlerPandas
from lightgbm_inference import LightGBMModelHandlerSciPy


class PostProcessor(beam.DoFn):
    def process(self, element: Tuple[int, PredictionResult]) -> Iterable[str]:
        label, prediction_result = element
        pred_proba = prediction_result.inference
        pred_label = numpy.argmax(pred_proba, axis=1)
        yield "{},{}".format(label, pred_label)


def parse_known_args(argv):
    """Parses args for the workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_type",
        dest="input_type",
        required=True,
        choices=["numpy", "pandas", "scipy"],
        help="Datatype of the input data.",
    )
    parser.add_argument(
        "--output",
        dest="output",
        required=True,
        help="Path to save output predictions.",
    )
    parser.add_argument(
        "--model_state",
        dest="model_state",
        required=True,
        help="Path to the state of the LightGBM model loaded for Inference.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", action="store_true", dest="split")
    group.add_argument("--no_split", action="store_false", dest="split")
    return parser.parse_known_args(argv)


def load_sklearn_iris_test_data(
    data_type: Callable, split: bool = True, seed: int = 999
) -> List[Union[numpy.array, pandas.DataFrame]]:
    dataset = load_iris()
    _, x_test, _, _ = train_test_split(
        dataset["data"], dataset["target"], test_size=0.2, random_state=seed
    )

    if split:
        return [
            (index, data_type(sample.reshape(1, -1)))
            for index, sample in enumerate(x_test)
        ]
    return [(0, data_type(x_test))]


def run(argv=None, save_main_session=True, test_pipeline=None) -> PipelineResult:
    known_args, pipeline_args = parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    data_types = {
        "numpy": (numpy.array, LightGBMModelHandlerNumpy),
        "pandas": (pandas.DataFrame, LightGBMModelHandlerPandas),
        "scipy": (scipy.sparse.csr_matrix, LightGBMModelHandlerSciPy),
    }

    input_data_type, model_handler = data_types[known_args.input_type]

    lightgbm_model_handler = KeyedModelHandler(
        model_handler(
            model_state=known_args.model_state
        )
    )

    input_data = load_sklearn_iris_test_data(
        data_type=input_data_type, split=known_args.split
    )

    pipeline = test_pipeline
    if not test_pipeline:
        pipeline = beam.Pipeline(options=pipeline_options)

    predictions = (
        pipeline
        | "ReadInputData" >> beam.Create(input_data)
        | "RunInference" >> RunInference(lightgbm_model_handler)
        | "PostProcessOutputs" >> beam.ParDo(PostProcessor())
    )

    _ = predictions | "WriteOutput" >> beam.io.WriteToText(
        known_args.output, shard_name_template="", append_trailing_newlines=True
    )

    result = pipeline.run()
    result.wait_until_finish()
    return result


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()