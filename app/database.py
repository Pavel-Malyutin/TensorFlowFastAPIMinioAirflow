import traceback

import pandas as pd
import sqlalchemy.engine
from sqlalchemy import create_engine, text

from app.config import get_settings

settings = get_settings()


class Database:
    def __init__(self):
        self.__settings = get_settings()
        self.__engine = self.__create_engine()

    def __create_engine(self) -> sqlalchemy.engine.Engine:
        user = self.__settings.pg_user
        port = self.__settings.db_port
        url = self.__settings.db_url
        database = self.__settings.database
        password = self.__settings.pg_password
        engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{url}:{port}/{database}")
        return engine

    def load_data(self, query) -> dict:
        try:
            df_from_records = pd.DataFrame(self.__engine.connect().execute(text(query)))
            result = {
                "data": df_from_records,
                "error": ""
            }
            return result
        except Exception as e:
            result = {
                "data": pd.DataFrame(),
                "error": f"Some error has occurred: {traceback.format_exc()}"
            }
            return result
        finally:
            self.__engine.dispose()

    def save_data(self, data: pd.DataFrame, table: str, drop: bool = True) -> str:
        try:
            if drop:
                connection = self.__engine.raw_connection()
                cursor = connection.cursor()
                command = f"DROP TABLE IF EXISTS {table};"
                cursor.execute(command)
                connection.commit()
                cursor.close()
            data.to_sql(table, self.__engine, if_exists='replace', index=False)
            return "Data uploaded"
        except Exception as e:
            return f"Some error has occurred: {traceback.format_exc()}"
        finally:
            self.__engine.dispose()
