import datetime as dt

from airflow.models import DAG
from airflow.operators.python_operator import PythonVirtualenvOperator

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=300),
    'depends_on_past': False,
}


def train(epochs):
    import datetime as dt
    import os

    import pandas as pd
    import tensorflow as tf

    from minio import Minio
    from sklearn.model_selection import train_test_split
    from sklearn.compose import make_column_transformer
    from sqlalchemy import create_engine, text
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Input

    db_user = os.environ['POSTGRES_USER']
    db_port = os.environ['PG_PORT']
    db_password = os.environ['POSTGRES_PASSWORD']
    database_name = os.environ['DATABASE_NAME']
    db_table = os.environ['TRAIN_TABLE']
    database_url = os.environ['PG_URL']

    engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{database_url}:{db_port}/{database_name}")

    columns = ['Unnamed: 0', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5', 'tag6', 'tag7',
               'tag8', 'tag9', 'tag10', 'tag11', 'tag12', 'tag13', 'tag14', 'tag15',
               'tag16', 'tag17', 'tag18', 'tag19', 'tag20', 'tag21', 'tag22', 'tag23',
               'tag24', 'tag25', 'tag26', 'tag27', 'tag28', 'tag29', 'tag30', 'tag31',
               'tag32', 'tag33', 'tag34', 'tag35', 'tag36', 'tag37', 'tag38', 'tag39',
               'tag40', 'tag41', 'tag42', 'tag43', 'tag44', 'tag45', 'tag46', 'tag47',
               'tag48', 'tag49', 'tag50', 'tag51', 'tag52', 'tag53', 'tag54', 'tag55',
               'tag56', 'tag57', 'tag58', 'tag59', 'tag60', 'tag61', 'tag62', 'tag63',
               'tag64', 'tag65', 'tag66', 'tag67', 'tag68', 'tag69', 'tag70', 'tag71',
               'tag72', 'tag73', 'tag74', 'tag75', 'tag76', 'tag77', 'tag78', 'tag79',
               'target1', 'target2', 'target3', 'target4']
    dataset = pd.DataFrame(engine.connect().execute(text(f"SELECT * FROM {db_table}")), columns=columns)
    engine.dispose()
    predict_col = ['target1', 'target2', 'target3', 'target4']
    columns_to_drop = ['Unnamed: 0']
    minmax_columns = [f"tag{i}" for i in range(1, 80)]
    if len(columns_to_drop) > 0:
        dataset = dataset.drop(columns_to_drop, axis=1)
    dataset = dataset.astype(float)
    dataset = dataset.interpolate(method='linear', axis=0).ffill().bfill()
    transformer = make_column_transformer(
        (MinMaxScaler(), minmax_columns)
    )
    features = dataset.drop(predict_col, axis=1)
    labels = dataset[predict_col]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    transformer.fit(train_features)
    train_features = transformer.transform(train_features)
    test_features = transformer.transform(test_features)

    def base_model(inputs):
        x = Dense(512, activation='relu')(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        return x

    def final_model(inputs):
        x = base_model(inputs)
        target1 = Dense(units='1', name='target1')(x)
        target2 = Dense(units='1', name='target2')(x)
        target3 = Dense(units='1', name='target3')(x)
        target4 = Dense(units='1', name='target4')(x)
        model = Model(inputs=inputs, outputs=[target1, target2, target3, target4])
        return model

    inputs = tf.keras.layers.Input(shape=(79,))

    model = final_model(inputs)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  loss={'target1': 'mse',
                        'target2': 'mse',
                        'target3': 'mse',
                        'target4': 'mse'},
                  metrics={
                      'target1': tf.keras.metrics.RootMeanSquaredError(),
                      'target2': tf.keras.metrics.RootMeanSquaredError(),
                      'target3': tf.keras.metrics.RootMeanSquaredError(),
                      'target4': tf.keras.metrics.RootMeanSquaredError(),
                  })
    model.fit(train_features, train_labels, epochs=epochs, validation_data=(test_features, test_labels))
    model_name = f"mymodel_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}.h5"
    model.save(model_name)

    client = Minio(
        os.environ['S3_ENDPOINT'],
        access_key=os.environ['MINIO_ROOT_USER'],
        secret_key=os.environ['MINIO_ROOT_PASSWORD'],
        secure=False
    )
    client.fput_object(os.environ['S3_BUCKET'], model_name, model_name)


with DAG(dag_id='train_model', default_args=args, schedule_interval=None) as dag:
    train_model = PythonVirtualenvOperator(
        task_id='train_model',
        python_callable=train,
        dag=dag,
        requirements=['scikit-learn', 'tensorflow', "minio"],
        op_kwargs={"epochs": 10}
    )
    train_model
