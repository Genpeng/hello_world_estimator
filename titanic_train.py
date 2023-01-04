# coding: utf-8

import tensorflow as tf

tf = tf.compat.v1
# if tf.__version__ >= "2.0":
#     tf = tf.compat.v1


def input_fn_builder(
    file_pattern,
    feature_spec,
    label=None,
    compression_type=None,
    num_epochs=1,
    batch_size=8,
    num_parallel_calls=8,
    shuffle_factor=0,
    prefetch_factor=10,
):
    def _parse(serialized):
        features = tf.io.parse_single_example(serialized, feature_spec)
        if label is not None:
            labels = features.pop(label)
            return features, tf.cast(labels, dtype=tf.int32)
        return features

    def input_fn():
        dataset = tf.data.Dataset.list_files(file_pattern)
        dataset = dataset.apply(
            tf.data.experimental.parallel_interleave(
                map_func=lambda filename: tf.data.TFRecordDataset(
                    filename, compression_type=compression_type
                ),
                cycle_length=num_parallel_calls,
                sloppy=True,
            )
        )
        dataset = dataset.map(_parse, num_parallel_calls=num_parallel_calls)

        if shuffle_factor > 0:
            dataset = dataset.shuffle(buffer_size=batch_size * shuffle_factor)

        dataset = dataset.repeat(num_epochs).batch(batch_size)

        if prefetch_factor > 0:
            dataset = dataset.prefetch(buffer_size=batch_size * prefetch_factor)

        # iterator = dataset.make_one_shot_iterator()
        # return iterator.get_next()

        return dataset

    return input_fn


def _get_titanic_feature_columns():
    categorical_columns = []

    passenger_id_col = tf.feature_column.categorical_column_with_identity(
        "PassengerId", num_buckets=1001
    )
    passenger_embed_col = tf.feature_column.embedding_column(
        passenger_id_col, dimension=10
    )
    categorical_columns.append(passenger_embed_col)

    p_class_col = tf.feature_column.categorical_column_with_vocabulary_list(
        "Pclass", vocabulary_list=[1, 2, 3]
    )
    p_class_onehot_col = tf.feature_column.indicator_column(p_class_col)
    categorical_columns.append(p_class_onehot_col)

    sex_col = tf.feature_column.categorical_column_with_vocabulary_list(
        "Sex", vocabulary_list=["male", "female"]
    )
    sex_onehot_col = tf.feature_column.indicator_column(sex_col)
    categorical_columns.append(sex_onehot_col)

    age_col = tf.feature_column.numeric_column("Age", dtype=tf.int64)
    age_onehot_col = tf.feature_column.bucketized_column(
        age_col, boundaries=[18, 22, 30, 40, 50, 60]
    )
    categorical_columns.append(age_onehot_col)

    embarked_col = tf.feature_column.categorical_column_with_vocabulary_list(
        "Embarked", vocabulary_list=["S", "C", "Q", "-1"]
    )
    embarked_onehot_col = tf.feature_column.indicator_column(embarked_col)
    categorical_columns.append(embarked_onehot_col)

    numerical_columns = []

    sibsp_col = tf.feature_column.numeric_column("SibSp", dtype=tf.int64)
    numerical_columns.append(sibsp_col)

    parch_col = tf.feature_column.numeric_column("Parch", dtype=tf.int64)
    numerical_columns.append(parch_col)

    return categorical_columns + numerical_columns


def main(unused_args):
    del unused_args

    train_file_pattern = "input/titanic/train/*.tfrecords"
    eval_file_pattern = "input/titanic/eval/*.tfrecords"
    num_epochs = 100
    train_batch_size = 32
    eval_batch_size = 32
    model_dir = "output/titanic_model"

    feature_spec = {
        "PassengerId": tf.io.FixedLenFeature([], tf.int64),
        "Survived": tf.io.FixedLenFeature([], tf.int64),
        "Pclass": tf.io.FixedLenFeature([], tf.int64),
        "Name": tf.io.FixedLenFeature([], tf.string),
        "Sex": tf.io.FixedLenFeature([], tf.string),
        "Age": tf.io.FixedLenFeature([], tf.float32),
        "SibSp": tf.io.FixedLenFeature([], tf.int64),
        "Parch": tf.io.FixedLenFeature([], tf.int64),
        "Ticket": tf.io.FixedLenFeature([], tf.string),
        "Fare": tf.io.FixedLenFeature([], tf.float32),
        "Cabin": tf.io.FixedLenFeature([], tf.string),
        "Embarked": tf.io.FixedLenFeature([], tf.string),
    }
    train_input_fn = input_fn_builder(
        train_file_pattern,
        feature_spec=feature_spec,
        label="Survived",
        num_epochs=num_epochs,
        batch_size=train_batch_size,
        shuffle_factor=10,
    )
    eval_input_fn = input_fn_builder(
        eval_file_pattern,
        feature_spec=feature_spec,
        label="Survived",
        num_epochs=1,
        batch_size=eval_batch_size,
    )

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[32, 16],
        feature_columns=_get_titanic_feature_columns(),
        model_dir=model_dir,
        n_classes=2,
        config=tf.estimator.RunConfig(
            save_checkpoints_secs=120,
            keep_checkpoint_max=5,
            save_summary_steps=100,
            log_step_count_steps=100,
        ),
    )
    # estimator = tf.estimator.Estimator(
    #     model_fn=model_fn_builder(),
    #     model_dir="xxx",
    #     config=tf.estimator.RunConfig(
    #         save_checkpoints_steps=None,
    #         save_checkpoints_secs=120,
    #         keep_checkpoint_max=5,
    #         save_summary_steps=100,
    #         log_step_count_steps=100,
    #     ),
    #     params={},
    # )
    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=10000,
            hooks=None,
        ),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=None,
            start_delay_secs=0,
            throttle_secs=0,  # 设置为 0 表示每次保存 checkpoints 时，都进行一次 evaluation
        ),
    )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
