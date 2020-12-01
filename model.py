import os
import json
import datagen
import config

import tensorflow as tf

from official import nlp
from official.nlp import bert

import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs


def save_model(classifier):
    export_dir = 'saved_model'
    tf.saved_model.save(classifier, export_dir=export_dir)


def test_model(classifier):
    tokenizer = config.tokenizer

    my_examples = datagen.bert_encode(
        data={
            'sentence1': [
                'The rain in Spain falls mainly on the plain.',
                'Look I fine tuned BERT.'],
            'sentence2': [
                'It mostly rains on the flat lands of Spain.',
                'Is it working? This does not match.']
        },
        tokenizer=tokenizer)

    result = classifier(my_examples, training=False)

    result = tf.argmax(result).numpy()
    print(result)


def bert_model():
    test, test_labels, train, train_labels, validation, validation_labels = datagen.prepare_data()

    # print(type(train_labels))
    print('##################################################### loading bert model...')

    bert_config_file = os.path.join(config.bert_folder, "bert_config.json")
    config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

    bert_config = bert.configs.BertConfig.from_dict(config_dict)

    print(config_dict)

    # binary classifier
    bert_classifier, bert_encoder = bert.bert_models.classifier_model(
        bert_config, num_labels=2)

    train_data_size = len(test_labels)
    steps_per_epoch = int(train_data_size / config.batch_size)
    num_train_steps = steps_per_epoch * config.epochs
    warmup_steps = int(config.epochs * train_data_size * 0.1 / config.batch_size)

    # creates an optimizer with learning rate schedule
    optimizer = nlp.optimization.create_optimizer(
        2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    # type(optimizer)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    """
    print(train_labels.items())
    count = 0
    for key, value in train_labels.items():
        print(type(value))
        count += 1
        if count == 10:
            break
    """
    print('##################################################### training model...')

    bert_classifier.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)

    # weird behavior here: tensorflow.python.framework.errors_impl.UnimplementedError:  Cast string to float is not supported
    #          [[node sparse_categorical_crossentropy/Cast (defined at /home/ebt/TESI/bert_nlp/model.py:79) ]] [Op:__inference_train_function_23235]
    bert_classifier.fit(
        train, train_labels,
        validation_data=(validation, validation_labels),
        batch_size=config.batch_size,
        epochs=config.epochs)

    # never tested
    print('##################################################### saving model...')
    save_model(bert_classifier)

    # only for dev purposes
    # print('##################################################### testing model...')
    test_model(bert_classifier)
    

