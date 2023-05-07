import sys
from abc import ABC
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union

import numpy
import pandas
import scipy

import lightgbm as lgb
from apache_beam.io.filesystems import FileSystems
from apache_beam.ml.inference.base import ExampleT
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import ModelT
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.ml.inference.base import PredictionT

__all__ = [
    "LightGBMModelHandler",
    "LightGBMModelHandlerNumpy",
    "LightGBMModelHandlerPandas",
    "LightGBMModelHandlerSciPy",
]

LightGBMInferenceFn = Callable[
    [
        Sequence[object],
        lgb.Booster,
        Optional[Dict[str, Any]],
    ],
    Iterable[PredictionResult],
]


def default_lightgbm_inference_fn(
    batch: Sequence[object],
    model: lgb.Booster,
    inference_args: Optional[Dict[str, Any]] = None,
) -> Iterable[PredictionResult]:
    inference_args = {} if not inference_args else inference_args

    predictions = [model.predict(el, **inference_args) for el in batch]

    return [PredictionResult(x, y) for x, y in zip(batch, predictions)]


class LightGBMModelHandler(ModelHandler[ExampleT, PredictionT, ModelT], ABC):
    def __init__(
        self,
        model_state: str,
        inference_fn: LightGBMInferenceFn = default_lightgbm_inference_fn,
    ):
        self._model_state = model_state
        self._inference_fn = inference_fn

    def load_model(self) -> lgb.Booster:
        with FileSystems.open(self._model_state, "rb") as model_state_file_handler:
            model_bin = model_state_file_handler.read()

        model_json_str = model_bin.decode("utf-8")

        booster = lgb.Booster(model_str=model_json_str)
        return booster

    def get_metrics_namespace(self) -> str:
        return "BeamML_LightGBM"


class LightGBMModelHandlerNumpy(
    LightGBMModelHandler[
        numpy.ndarray, PredictionResult, lgb.Booster
    ]
):
    def run_inference(
        self,
        batch: Sequence[numpy.ndarray],
        model: lgb.Booster,
        inference_args: Optional[Dict[str, Any]] = None,
    ) -> Iterable[PredictionResult]:
        return self._inference_fn(batch, model, inference_args)

    def get_num_bytes(self, batch: Sequence[numpy.ndarray]) -> int:
        return sum(sys.getsizeof(element) for element in batch)


class LightGBMModelHandlerPandas(
    LightGBMModelHandler[
        pandas.DataFrame, PredictionResult, Union[lgb.Booster, lgb.LGBMModel]
    ]
):
    def run_inference(
        self,
        batch: Sequence[pandas.DataFrame],
        model: lgb.Booster,
        inference_args: Optional[Dict[str, Any]] = None,
    ) -> Iterable[PredictionResult]:
        return self._inference_fn(batch, model, inference_args)

    def get_num_bytes(self, batch: Sequence[pandas.DataFrame]) -> int:
        return sum(df.memory_usage(deep=True).sum() for df in batch)


class LightGBMModelHandlerSciPy(
    LightGBMModelHandler[
        scipy.sparse.csr_matrix,
        PredictionResult,
        lgb.Booster
    ]
):
    def run_inference(
        self,
        batch: Sequence[scipy.sparse.csr_matrix],
        model: lgb.Booster,
        inference_args: Optional[Dict[str, Any]] = None,
    ) -> Iterable[PredictionResult]:
        return self._inference_fn(batch, model, inference_args)

    def get_num_bytes(self, batch: Sequence[scipy.sparse.csr_matrix]) -> int:
        return sum(sys.getsizeof(element) for element in batch)