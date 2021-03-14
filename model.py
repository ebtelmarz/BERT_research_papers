import os
import json
import numpy as np
import matplotlib.pyplot as plt

import datagen
import config

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    checkpoint = tf.train.Checkpoint(model=bert_encoder)
    checkpoint.restore(
        os.path.join(config.bert_folder, 'bert_model.ckpt'))

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
                               patience=2,  # number of epochs
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
    history = classifier.fit(train,
                             train_labels,
                             validation_data=(validation, validation_labels),
                             batch_size=config.batch_size,
                             epochs=config.epochs,
                             callbacks=[early_exit])

    return classifier, history


def show_confusion_matrix(true, predictions):
    matrix = confusion_matrix(true, predictions)
    rel_matrix = matrix / np.sum(matrix, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(20, 40))

    image1 = axes[0].imshow(matrix, cmap=plt.get_cmap('GnBu'))

    for (i, j), e in np.ndenumerate(matrix):
        axes[0].text(j, i, s=str(e), ha='center', va='center')

    axes[0].set_xticks(np.arange(0, len(config.class_labels), 1))
    axes[0].set_xticklabels(config.class_labels)
    axes[0].set_yticks(np.arange(0, len(config.class_labels), 1))
    axes[0].set_yticklabels(config.class_labels)
    axes[0].set_title('Confusion Matrix')

    image2 = axes[1].imshow(matrix / np.sum(matrix, axis=0), cmap=plt.get_cmap('GnBu'))

    for (i, j), e in np.ndenumerate(rel_matrix):
        axes[1].text(j, i, s=str(np.round(e, 2)), ha='center', va='center')

    axes[1].set_xticks(np.arange(0, len(config.class_labels), 1))
    axes[1].set_xticklabels(config.class_labels)
    axes[1].set_yticks(np.arange(0, len(config.class_labels), 1))
    axes[1].set_yticklabels(config.class_labels)
    plt.subplots_adjust(wspace=0.5)
    axes[1].set_title('Confusion Matrix (Relative)')

    plt.savefig('matrix.png', bbox_inches='tight')


def do_classification_report(true, predictions):
    confusion = confusion_matrix(true, predictions)
    print('Confusion Matrix\n')
    print(confusion)

    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(true, predictions)))

    print('Micro Precision: {:.2f}'.format(precision_score(true, predictions, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(true, predictions, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(true, predictions, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(true, predictions, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(true, predictions, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(true, predictions, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(true, predictions, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(true, predictions, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(true, predictions, average='weighted')))

    print('\nClassification Report\n')
    print(classification_report(true, predictions, target_names=config.class_labels))


def show_metrics(hst):
    history_dict = hst.history
    loss_values = history_dict['loss']
    validation_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    validation_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(30, 10))
    training_ts = [loss_values, acc_values]
    validation_ts = [validation_loss_values, validation_acc_values]
    metric_names = ['loss', 'categorical accuracy']

    for i in range(len(axes)):
        axes[i].plot(epochs, training_ts[i], color='r', label='training')
        axes[i].plot(epochs, validation_ts[i], color='b', label='validation')
        axes[i].set_xlabel('epoch')
        axes[i].set_ylabel(metric_names[i])
        axes[i].set_title(metric_names[i] + ' analysis')
        axes[i].set_xticks(np.arange(0, epochs[-1] + 1, 5))
        axes[i].set_yticks(np.arange(0, 1.1, 0.1))
        axes[i].set_xlim([1, epochs[-1]])
        axes[i].set_ylim([np.min([np.min(training_ts[i]), np.min(validation_ts[i])]),
                          np.max([np.max(training_ts[i]), np.max(validation_ts[i])])])
        axes[i].legend()

        plt.savefig('metrics.png', bbox_inches='tight')


def predict_test_set(model, test, test_labels):
    prediction = model.predict(test, batch_size=config.batch_size).argmax(axis=-1)
    evaluation = model.evaluate(x=test,
                                y=test_labels,
                                batch_size=config.batch_size)
    print(prediction)
    print(evaluation)

    return prediction


def bert_model():
    # prepare data
    test, test_labels, train, train_labels, validation, validation_labels = datagen.prepare_data()

    # construct the model
    classifier, encoder = create_model()

    # fit the model
    fitted_bert, history = fit_bert(classifier, test_labels, train, train_labels, validation, validation_labels)

    print('\n##################################################### saving model...')
    save_model(fitted_bert)

    # make prediction
    prediction = predict_test_set(fitted_bert, test, test_labels)

    # print metrics, confusion matrix and classification report
    show_confusion_matrix(test_labels, prediction)
    show_metrics(history)
    do_classification_report(test_labels, prediction)

    return fitted_bert, history
