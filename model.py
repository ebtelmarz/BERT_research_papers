import os
import json
import datagen
import config

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs


def save_model(classifier):
    tf.saved_model.save(classifier, export_dir=config.export_dir)

"""
def test_model(classifier, other):

    ## line count in parse_row does not work with a tail
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
    
    # real test
    os.system('tail -100 data/raw-data.csv > data/raw-data_tail.csv')
    datagen.do_datagen(config.input_test_raw_data, config.test_whole_dataset, other)
    os.system('shuf data/raw-data_tail.csv -o shuffled/shuf_test_dataset.csv')

    test_df = pd.read_csv('shuffled/shuf_test_dataset.csv', sep=',', names=config.header)

    test = datagen.bert_encode(test_df, tokenizer)
    # test_labels_df = test_df['score'].map(lambda x: x != 'score')
    test_labels = test_df['score'].astype('int64')

    if classifier:
        evaluation = classifier.evaluate(x=test,
                                         y=test_labels,
                                         batch_size=config.batch_size)
        print(evaluation)

    else:
        reloaded = tf.saved_model.load(config.export_dir)
        evaluation = reloaded.evaluate(x=test,
                                       y=test_labels,
                                       batch_size=config.batch_size)
        print(evaluation)
"""


def create_model():
    print('\n##################################################### loading bert model...')

    bert_config_file = os.path.join(config.bert_folder, "bert_config.json")
    config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

    bert_config = bert.configs.BertConfig.from_dict(config_dict)

    print(config_dict)

    # binary classifier
    bert_classifier, bert_encoder = bert.bert_models.classifier_model(
        bert_config, num_labels=2)

    return bert_classifier, bert_encoder


def fit_bert(classifier, labels, train, train_labels, validation, validation_labels):
    train_data_size = len(labels)
    steps_per_epoch = int(train_data_size / config.batch_size)
    num_train_steps = steps_per_epoch * config.epochs
    warmup_steps = int(config.epochs * train_data_size * 0.1 / config.batch_size)

    # creates an optimizer with learning rate schedule
    optimizer = nlp.optimization.create_optimizer(2e-5,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=warmup_steps)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    early_exit = EarlyStopping(monitor='val_loss',
                               patience=1,  # number of epochs
                               verbose=1,
                               mode='min')

    best_checkpoint = ModelCheckpoint('.best_fit.hdf5',
                                      save_best_only=True,
                                      monitor='val_categorical_accuracy',
                                      mode='max')

    print('\n##################################################### training model...')
    classifier.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)

    # [SOLVED] weird behavior here: tensorflow.python.framework.errors_impl.UnimplementedError:  Cast string to float is not supported
    #          [[node sparse_categorical_crossentropy/Cast (defined at /home/ebt/TESI/bert_nlp/model.py:79) ]] [Op:__inference_train_function_23235]
    classifier.fit(train,
                   train_labels,
                   validation_data=(validation, validation_labels),
                   batch_size=config.batch_size,
                   epochs=config.epochs,
                   callbacks=[early_exit])

    return classifier


def predict_test_set(model, test, test_labels):
    prediction = model.predict(test, batch_size=config.batch_size)
    evaluation = model.evaluate(x=test,
                                y=test_labels,
                                batch_size=config.batch_size)
    print(prediction)
    print(evaluation)


def bert_model():
    # prepare data
    test, test_labels, train, train_labels, validation, validation_labels = datagen.prepare_data()

    # construct the model
    classifier, encoder = create_model()

    # fit the model
    fitted_bert = fit_bert(classifier, test_labels, train, train_labels, validation, validation_labels)

    print('\n##################################################### saving model...')
    save_model(fitted_bert)

    # make prediction
    predict_test_set(fitted_bert, test, test_labels)

    return fitted_bert
