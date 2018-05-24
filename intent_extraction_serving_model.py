import spacy
import numpy as np
from keras import Input, Model
from keras.layers import Embedding, Dropout, TimeDistributed, Bidirectional, LSTM, concatenate, \
    Dense
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from nlp_architect.data.intent_datasets import SNIPS

sentence_length = 50
word_length = 12
dataset_path = '/Users/pizsak/nlp_architect/data/snips/'
dataset = SNIPS(path=dataset_path,
                sentence_length=sentence_length,
                word_length=word_length)


def load_model():
    embedding_size = 100
    word_vocabulary_size = dataset.vocab_size
    character_vocab_size = dataset.char_vocab_size
    character_emb_size = 25
    char_lstm_hidden_dims = 25
    intent_labels = dataset.intent_size
    tagging_lstm_hidden_dims = 100
    tag_labels = dataset.label_vocab_size
    words_input = Input(shape=(sentence_length,), name='words_input')
    embedding_layer = Embedding(word_vocabulary_size, embedding_size,
                                input_length=sentence_length, trainable=True,
                                name='word_embedding_layer')
    word_embeddings = embedding_layer(words_input)
    word_embeddings = Dropout(0.5)(word_embeddings)
    word_chars_input = Input(shape=(sentence_length, word_length), name='word_chars_input')
    char_embedding_layer = Embedding(character_vocab_size, character_emb_size,
                                     input_length=word_length, name='char_embedding_layer')
    char_embeddings = TimeDistributed(char_embedding_layer)(word_chars_input)
    char_embeddings = TimeDistributed(Bidirectional(LSTM(char_lstm_hidden_dims)))(char_embeddings)
    char_embeddings = Dropout(0.5)(char_embeddings)
    shared_bilstm_layer = Bidirectional(LSTM(100, return_sequences=True, return_state=True))
    shared_lstm_out = shared_bilstm_layer(word_embeddings)
    shared_lstm_y = shared_lstm_out[:1][0]  # save y states of the LSTM layer
    states = shared_lstm_out[1:]
    hf, cf, hb, cb = states  # extract last hidden states
    h_state = concatenate([hf, hb], axis=-1)  # concatenate last states
    intent_out = Dense(intent_labels, activation='softmax',
                       name='intent_classifier_output')(h_state)
    combined_features = concatenate([shared_lstm_y, char_embeddings], axis=-1)
    tagging_lstm = Bidirectional(LSTM(tagging_lstm_hidden_dims, return_sequences=True))(
            combined_features)
    second_bilstm_layer = Dropout(0.5)(tagging_lstm)
    crf = CRF(tag_labels, sparse_target=False)
    labels_out = crf(second_bilstm_layer)
    model = Model(inputs=[words_input, word_chars_input],
                  outputs=[intent_out, labels_out])
    loss_f = {'intent_classifier_output': 'categorical_crossentropy',
              'crf_1': crf.loss_function}
    metrics = {'intent_classifier_output': 'categorical_accuracy',
               'crf_1': crf.accuracy}
    model.compile(loss=loss_f,
                  optimizer='adam',
                  metrics=metrics)
    model_name = 'my_model'
    print('Loading model weights')
    save_load_utils.load_all_weights(model, model_name, include_optimizer=False)
    return model


nlp = spacy.load('en')


def encode_word(word):
    return dataset.tokens_vocab.get(word, 1.0)


def encode_word_chars(word):
    return [dataset._chars_vocab.vocab.get(c, 1.0) for c in word]


def process_text(text):
    input_text = ' '.join(text.strip().split())
    doc = nlp(input_text)
    return [t.text for t in doc]


def encode_input(text_arr):
    sentence = []
    sentence_chars = []
    for word in text_arr:
        sentence.append(encode_word(word))
        sentence_chars.append(encode_word_chars(word))
    encoded_sentence = pad_sequences([np.asarray(sentence)], maxlen=sentence_length)
    chars_padded = pad_sequences(sentence_chars, maxlen=word_length)
    if sentence_length - chars_padded.shape[0] > 0:
        chars_padded = np.concatenate((np.zeros((sentence_length - chars_padded.shape[0],
                                                 word_length)), chars_padded))
    encoded_chars = chars_padded.reshape(1, sentence_length, word_length)
    return encoded_sentence, encoded_chars


def detect_ents(text, tags):
    ents = []
    in_ent = False
    s = None
    e = None
    ent_type = None
    for i, t in enumerate(tags):
        if t != '0' and t is not None and t.startswith('B-'):
            if in_ent:
                ents.append({'start': s,
                             'end': e,
                             'type': ent_type})
            s = len(' '.join(text[:i])) + 1
            e = s + len(text[i])
            ent_type = t.split('-')[1]
            in_ent = True
        elif in_ent is True and t.startswith('I-'):
            e += 1 + len(text[i])
        else:
            if in_ent:
                ents.append({'start': s,
                             'end': e,
                             'type': ent_type})
            in_ent = False
    return ents


def serve():
    model = load_model()

    def process(text):
        text_arr = process_text(text)
        words, chars = encode_input(text_arr)
        intent, tags = model.predict([words, chars])
        intent = intent.argmax()
        intent_str = dataset._intents_vocab.id_to_word(intent)
        tags = tags.argmax(2)
        tags_str = [dataset._tags_vocab.id_to_word(t) for t in tags[0]][-len(text):][
                   -len(text_arr):]
        text = ' '.join(text_arr)
        ents = detect_ents(text_arr, tags_str)
        res = {'text': text,
               'ents': ents,
               'title': None}
        return intent_str, res

    return process
