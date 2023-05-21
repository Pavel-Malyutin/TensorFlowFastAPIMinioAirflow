from __future__ import annotations

import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import tensorflow as tf
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler

from app.config import get_settings
from app.database import Database
from minio import Minio

settings = get_settings()


@dataclass
class Request:
    errors: str = ""
    data: pd.DataFrame = None
    prepared_data: Any = None
    model: Any = None
    status: str = None
    predictions: pd.DataFrame = None


def load_dataset_predict(dataset: pd.DataFrame, columns_to_drop: list, minmax_columns: list) -> tuple:
    if len(columns_to_drop) > 0:
        dataset = dataset.drop(columns_to_drop, axis=1)
    dataset = dataset.astype(float)
    dataset = dataset.interpolate(method='linear', axis=0).ffill().bfill()
    dataset = dataset.fillna(0)
    transformer = make_column_transformer(
        (MinMaxScaler(), minmax_columns)
    )
    transformer.fit(dataset)
    dataset = transformer.transform(dataset)
    return dataset


class Handler(ABC):
    @abstractmethod
    def set_next(self, handler: Handler) -> Handler:
        pass

    @abstractmethod
    def handle(self, request) -> Optional[Request]:
        pass


class AbstractHandler(Handler):
    _next_handler: Handler = None

    def set_next(self, handler: Handler) -> Handler:
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, request: Any) -> Request:
        if self._next_handler:
            return self._next_handler.handle(request)
        return None


class DataLoader(AbstractHandler):
    def handle(self, request: Request) -> Request:
        if len(request.errors) == 0:
            try:
                database = Database()
                query = f"SELECT * FROM {settings.predict_table}"
                result = database.load_data(query)
                request.data = result["data"]
                request.errors = result["error"]
            except:
                request.errors = f"Error in DataLoader: {traceback.format_exc()}"
            return super().handle(request)
        else:
            return super().handle(request)


class ModelLoader(AbstractHandler):
    def handle(self, request: Request) -> Request:
        if len(request.errors) == 0:
            try:
                client = Minio(
                    settings.minio_url,
                    access_key=settings.minio_user,
                    secret_key=settings.minio_password,
                    secure=False
                )
                objects = client.list_objects(settings.minio_bucket)
                objects = [(i, i.last_modified) for i in objects]
                objects = sorted(objects, key=lambda obj: obj[1], reverse=True)
                last_object = objects[0][0]
                client.fget_object(settings.minio_bucket, last_object.object_name, "temp.h5")
                request.model = tf.keras.models.load_model("temp.h5")
            except:
                request.errors = f"Error in ModelLoader: {traceback.format_exc()}"
            return super().handle(request)
        else:
            return super().handle(request)


class DataPreparer(AbstractHandler):
    def handle(self, request: Request) -> Request:
        if len(request.errors) == 0:
            try:
                request.prepared_data = load_dataset_predict(
                    dataset=request.data,
                    columns_to_drop=['Unnamed: 0'],
                    minmax_columns=['tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6', 'tag7', 'tag8', 'tag9', 'tag10',
                                    'tag11', 'tag12', 'tag13', 'tag14', 'tag15', 'tag16', 'tag17', 'tag18', 'tag19',
                                    'tag20', 'tag21', 'tag22', 'tag23', 'tag24', 'tag25', 'tag26', 'tag27', 'tag28',
                                    'tag29', 'tag30', 'tag31', 'tag32', 'tag33', 'tag34', 'tag35', 'tag36', 'tag37',
                                    'tag38', 'tag39', 'tag40', 'tag41', 'tag42', 'tag43', 'tag44', 'tag45', 'tag46',
                                    'tag47', 'tag48', 'tag49', 'tag50', 'tag51', 'tag52', 'tag53', 'tag54', 'tag55',
                                    'tag56', 'tag57', 'tag58', 'tag59', 'tag60', 'tag61', 'tag62', 'tag63', 'tag64',
                                    'tag65', 'tag66', 'tag67', 'tag68', 'tag69', 'tag70', 'tag71', 'tag72', 'tag73',
                                    'tag74', 'tag75', 'tag76', 'tag77', 'tag78', 'tag79']
                                    )
            except:
                request.errors = f"Error in DataPreparer: {traceback.format_exc()}"
            return super().handle(request)
        else:
            return super().handle(request)


class Predictor(AbstractHandler):
    def handle(self, request: Request) -> Request:
        if len(request.errors) == 0:
            try:
                request.predictions = request.model.predict(request.prepared_data)
            except:
                request.errors = f"Error in Predictor: {traceback.format_exc()}"
            return super().handle(request)
        else:
            return super().handle(request)


class DataSaver(AbstractHandler):
    def handle(self, request: Request) -> Request:
        if len(request.errors) == 0:
            uploader = Database()
            for i, n in enumerate(['target1', 'target2', 'target3', 'target4']):
                request.data[n] = request.predictions[i]
            status = uploader.save_data(request.data, table=settings.result_table)
            if status == "Data uploaded":
                request.status = "Predictions saved"
            else:
                request.errors = status
            return super().handle(request)
        else:
            return super().handle(request)


def get_predictions() -> str:
    data = DataLoader()
    model = ModelLoader()
    prepared_data = DataPreparer()
    predictions = Predictor()
    saver = DataSaver()
    request = Request()
    data.set_next(model).set_next(prepared_data).set_next(predictions).set_next(saver)
    data.handle(request)
    if len(request.errors) > 0:
        return request.errors
    else:
        return request.status
