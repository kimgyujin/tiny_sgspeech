import collections
import tensorflow as tf

from . import Model
from ..utils.utils import get_rnn, shape_list, count_non_blank, pad_prediction_tfarray
from ..featurizers.speech_featurizer import SpeechFeaturizer
from ..featurizers.text_featurizer import TextFeaturizer
from .layers.embedding import Embedding

Hypothesis = collections.namedtuple("Hypothesis", ("index", "prediction", "states"))

BeamHypothesis = collections.namedtuple("BeamHypothesis", ("score", "indices", "prediction", "states"))

class TransducerPrediction(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 embed_dim: int,
                 embed_dropout: float = 0,
                 num_rnns: int = 1,
                 rnn_units: int = 512,
                 rnn_type: str = "lstm",
                 rnn_implementation: int = 2,
                 layer_norm: bool = True,
                 projection_units: int = 0,
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 name="transducer_prediction",
                 **kwargs):
        super(TransducerPrediction, self).__init__(name=name, **kwargs)
        self.embed = Embedding(vocabulary_size, embed_dim,
                               regularizer=kernel_regularizer, name=f"{name}_embedding")
        self.do = tf.keras.layers.Dropout(embed_dropout, name=f"{name}_dropout")

        RNN=get_rnn(rnn_type)
        self.rnns = []
        for i in range(num_rnns):
            rnn = RNN(
                units=rnn_units, return_sequences=True,
                name=f"{name}_{rnn_type}_{i}", return_state=True,
                implementation=rnn_implementation,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
            if layer_norm:
                ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln_{i}")
            else:
                ln = None
            if projection_units > 0:
                projection = tf.keras.layers.Dense(
                    projection_units,
                    name=f"{name}_projection_{i}",
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer
                )
            else:
                projection = None
            self.rnns.append({"rnn": rnn, "ln":ln, "projection": projection})

    def get_initial_state(self):
        states = []
        for rnn in self.rnns:
            states.append(
                tf.stack(
                    rnn["rnn"].get_initial_state(
                        tf.zeros([1,1,1], dtype=tf.float32)
                    ), axis=0
                )
            )
        return tf.stack(states, axis=0)

    def call(self, inputs, training=False, **kwargs):

        outputs, prediction_length = inputs
        outputs = self.embed(outputs, training=training)
        outputs = self.do(outputs, training=training)
        for rnn in self.rnns:
            mask = tf.sequence_mask(prediction_length, maxlen=tf.shape(outputs)[1])
            outputs = rnn["rnn"](outputs, training=training, mask=mask)
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=training)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=training)
        return outputs

    def recognize(self, inputs, states):
        """Recognize function for prediction network
        Args:
            inputs (tf.Tensor): shape [1, 1]
            states (tf.Tensor): shape [num_lstms, 2, B, P]
        Returns:
            tf.Tensor: outputs with shape [1, 1, P]
            tf.Tensor: new states with shape [num_lstms, 2, 1, P]
        """
        outputs = self.embed(inputs, training=False)
        outputs = self.do(outputs, training=False)
        new_states = []
        for i, rnn in enumerate(self.rnns):
            outputs = rnn["rnn"](outputs, training=False, initial_state=tf.unstack(states[i], axis=0))
            new_states.append(tf.stack(outputs[1:]))
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=False)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=False)
        return outputs, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.embed.get_config()
        conf.update(self.do.get_config())
        for rnn in self.rnns:
            conf.update(rnn["rnn"].get_config())
            if rnn["ln"] is not None:
                conf.update(rnn["ln"].get_config())
            if rnn["projection"] is not None:
                conf.update(rnn["projection"].get_config())
        return conf


class TransducerJoint(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 joint_dim: int = 1024,
                 activation: str = "tanh",
                 prejoint_linear: bool = True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="tranducer_joint",
                 **kwargs):
        super(TransducerJoint, self).__init__(name=name, **kwargs)

        activation = activation.lower()
        if activation == "linear": self.activation = tf.keras.activation.linear
        elif activation == "relu": self.activation = tf.nn.relu
        elif activation == "tanh": self.activation = tf.nn.tanh
        else: raise ValueError("activation must be either 'linear', 'relu' or 'tanh'")

        self.prejoint_linear = prejoint_linear

        if self.prejoint_linear:
            self.ffn_enc = tf.keras.layers.Dense(
                joint_dim, name=f"{name}_enc",
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
            self.ffn_pred = tf.keras.layers.Dense(
                joint_dim, use_bias=False, name=f"{name}_pred",
                kernel_regularizer=kernel_regularizer
            )
        self.ffn_out = tf.keras.layers.Dense(
            vocabulary_size, name=f"{name}_vocab",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

    def call(self, inputs, training=False, **kwargs):
        # enc has shape [B, T, E]
        # pred has shape [B, U, P]
        enc_out, pred_out = inputs
        if self.prejoint_linear:
            enc_out = self.ffn_enc(enc_out, training=training)  # [B, T, E] => [B, T, V]
            pred_out = self.ffn_pred(pred_out, training=training)  # [B, U, P] => [B, U, V]
        enc_out = tf.expand_dims(enc_out, axis=2)
        pred_out = tf.expand_dims(pred_out, axis=1)
        outputs = self.activation(enc_out + pred_out)  # => [B, T, U, V]
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def get_config(self):
        conf = self.ffn_enc.get_config()
        conf.update(self.ffn_pred.get_config())
        conf.update(self.ffn_out.get_config())
        conf.update({"prejoint_linear": self.prejoint_linear, "activation": self.activation})
        return conf


class Transducer(Model):
    """ Transducer Model Warper """

    def __init__(self,
                 encoder: tf.keras.Model,
                 vocabulary_size: int,
                 embed_dim: int = 512,
                 embed_dropout: float = 0,
                 num_rnns: int = 1,
                 rnn_units: int = 320,
                 rnn_type: str = "lstm",
                 rnn_implementation: int = 2,
                 layer_norm: bool = True,
                 projection_units: int = 0,
                 joint_dim: int = 1024,
                 joint_activation: str = "tanh",
                 prejoint_linear: bool = True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name="transducer",
                 **kwargs):
        super(Transducer, self).__init__(name=name, **kwargs)
        self.encoder = encoder
        self.predict_net = TransducerPrediction(
            vocabulary_size=vocabulary_size,
            embed_dim=embed_dim,
            embed_dropout=embed_dropout,
            num_rnns=num_rnns,
            rnn_units=rnn_units,
            rnn_type=rnn_type,
            rnn_implementation=rnn_implementation,
            layer_norm=layer_norm,
            projection_units=projection_units,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_prediction"
        )
        self.joint_net = TransducerJoint(
            vocabulary_size=vocabulary_size,
            joint_dim=joint_dim,
            activation=joint_activation,
            prejoint_linear=prejoint_linear,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}_joint"
        )
        self.time_reduction_factor = 1

    def _build(self, input_shape, prediction_shape=[None], batch_size=None):
        inputs = tf.keras.Input(shape=input_shape, batch_size=batch_size, dtype=tf.float32)
        input_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        pred = tf.keras.Input(shape=prediction_shape, batch_size=batch_size, dtype=tf.int32)
        pred_length = tf.keras.Input(shape=[], batch_size=batch_size, dtype=tf.int32)
        self([inputs, input_length, pred, pred_length], training=False)

    def summary(self, line_length=None, **kwargs):
        if self.encoder is not None: self.encoder.summary(line_length=line_length, **kwargs)
        self.predict_net.summary(line_length=line_length, **kwargs)
        self.joint_net.summary(line_length=line_length, **kwargs)
        super(Transducer, self).summary(line_length=line_length, **kwargs)

    def add_featurizers(self,
                        speech_featurizer: SpeechFeaturizer,
                        text_featurizer: TextFeaturizer):
        """
        Function to add featurizer to model to convert to end2end tfliteggg
        Args:
            speech_featurizer: SpeechFeaturizer instance
            text_featurizer: TextFeaturizer instance
            scorer: external language model scorer
        """
        self.speech_featurizer = speech_featurizer
        self.text_featurizer = text_featurizer

    def call(self, inputs, training=False, **kwargs):
        """
        Transducer Model call function
        Args:
            features: audio features in shape [B, T, F, C]
            input_length: features time length in shape [B]
            prediction: predicted sequence of ids, in shape [B, U]
            prediction_length: predicted sequence of ids length in shape [B]
            training: python boolean
            **kwargs: sth else
        Returns:
            `logits` with shape [B, T, U, vocab]
        """
        features, _, prediction, prediction_length = inputs
        enc = self.encoder(features, training=training, **kwargs)
        pred = self.predict_net([prediction, prediction_length], training=training, **kwargs)
        outputs = self.joint_net([enc, pred], training=training, **kwargs)
        return outputs

    # -------------------------------- INFERENCES-------------------------------------

    def encoder_inference(self, features: tf.Tensor):
        """Infer function for encoder (or encoders)
        Args:
            features (tf.Tensor): features with shape [T, F, C]
        Returns:
            tf.Tensor: output of encoders with shape [T, E]
        """
        with tf.name_scope(f"{self.name}_encoder"):
            outputs = tf.expand_dims(features, axis=0)
            outputs = self.encoder(outputs, training=False)
            return tf.squeeze(outputs, axis=0)

    def decoder_inference(self, encoded: tf.Tensor, predicted: tf.Tensor, states: tf.Tensor):
        """Infer function for decoder
        Args:
            encoded (tf.Tensor): output of encoder at each time step => shape [E]
            predicted (tf.Tensor): last character index of predicted sequence => shape []
            states (nested lists of tf.Tensor): states returned by rnn layers
        Returns:
            (ytu, new_states)
        """
        with tf.name_scope(f"{self.name}_decoder"):
            encoded = tf.reshape(encoded, [1, 1, -1])  # [E] => [1, 1, E]
            predicted = tf.reshape(predicted, [1, 1])  # [] => [1, 1]
            y, new_states = self.predict_net.recognize(predicted, states)  # [1, 1, P], states
            ytu = tf.nn.log_softmax(self.joint_net([encoded, y], training=False))  # [1, 1, V]
            ytu = tf.reshape(ytu, shape=[-1])  # [1, 1, V] => [V]
            return ytu, new_states

    def get_config(self):
        conf = self.encoder.get_config()
        conf.update(self.predict_net.get_config())
        conf.update(self.joint_net.get_config())
        return conf

    # -------------------------------- GREEDY -------------------------------------

    @tf.function
    def recognize(self,
                  features: tf.Tensor,
                  input_length: tf.Tensor,
                  parallel_iterations: int = 10,
                  swap_memory: bool = True):
        """
        RNN Transducer Greedy decoding
        Args:
            features (tf.Tensor): a batch of extracted features
            input_length (tf.Tensor): a batch of extracted features length
        Returns:
            tf.Tensor: a batch of decoded transcripts
        """
        encoded = self.encoder(features, training=False)
        return self._perform_greedy_batch(encoded, input_length,
                                          parallel_iterations=parallel_iterations, swap_memory=swap_memory)

    def _perform_greedy_batch(self,
                              encoded: tf.Tensor,
                              encoded_length: tf.Tensor,
                              parallel_iterations: int = 10,
                              swap_memory: bool = False,
                              version: str = 'v1'):
        with tf.name_scope(f"{self.name}_perform_greedy_batch"):
            total_batch = tf.shape(encoded)[0]
            batch = tf.constant(0, dtype=tf.int32)

            greedy_fn = self._perform_greedy if version == 'v1' else self._perform_greedy_v2

            decoded = tf.TensorArray(
                dtype=tf.int32, size=total_batch, dynamic_size=False,
                clear_after_read=False, element_shape=tf.TensorShape([None])
            )

            def condition(batch, _): return tf.less(batch, total_batch)

            def body(batch, decoded):
                hypothesis = greedy_fn(
                    encoded=encoded[batch],
                    encoded_length=encoded_length[batch],
                    predicted=tf.constant(self.text_featurizer.blank, dtype=tf.int32),
                    states=self.predict_net.get_initial_state(),
                    parallel_iterations=parallel_iterations,
                    swap_memory=swap_memory
                )
                decoded = decoded.write(batch, hypothesis.prediction)
                return batch + 1, decoded

            batch, decoded = tf.while_loop(
                condition, body, loop_vars=[batch, decoded],
                parallel_iterations=parallel_iterations, swap_memory=True,
            )

            decoded = pad_prediction_tfarray(decoded, blank=self.text_featurizer.blank)
            return self.text_featurizer.iextract(decoded.stack())

    def _perform_greedy(self,
                        encoded: tf.Tensor,
                        encoded_length: tf.Tensor,
                        predicted: tf.Tensor,
                        states: tf.Tensor,
                        parallel_iterations: int = 10,
                        swap_memory: bool = False):
        with tf.name_scope(f"{self.name}_greedy"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32, size=total, dynamic_size=False,
                    clear_after_read=False, element_shape=tf.TensorShape([])
                ),
                states=states
            )

            def condition(_time, _hypothesis): return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                # something is wrong with tflite that drop support for tf.cond
                # def equal_blank_fn(): return _hypothesis.index, _hypothesis.states
                # def non_equal_blank_fn(): return _predict, _states  # update if the new prediction is a non-blank
                # _index, _states = tf.cond(tf.equal(_predict, blank), equal_blank_fn, non_equal_blank_fn)

                _equal = tf.equal(_predict, self.text_featurizer.blank)
                _index = tf.where(_equal, _hypothesis.index, _predict)
                _states = tf.where(_equal, _hypothesis.states, _states)

                _prediction = _hypothesis.prediction.write(_time, _predict)
                _hypothesis = Hypothesis(index=_index, prediction=_prediction, states=_states)

                return _time + 1, _hypothesis

            time, hypothesis = tf.while_loop(
                condition, body,
                loop_vars=[time, hypothesis], parallel_iterations=parallel_iterations, swap_memory=swap_memory
            )

            return Hypothesis(index=hypothesis.index, prediction=hypothesis.prediction.stack(), states=hypothesis.states)

    def _perform_greedy_v2(self,
                           encoded: tf.Tensor,
                           encoded_length: tf.Tensor,
                           predicted: tf.Tensor,
                           states: tf.Tensor,
                           parallel_iterations: int = 10,
                           swap_memory: bool = False):
        """ Ref: https://arxiv.org/pdf/1801.00841.pdf """
        with tf.name_scope(f"{self.name}_greedy_v2"):
            time = tf.constant(0, dtype=tf.int32)
            total = encoded_length

            hypothesis = Hypothesis(
                index=predicted,
                prediction=tf.TensorArray(
                    dtype=tf.int32, size=0, dynamic_size=True,
                    clear_after_read=False, element_shape=tf.TensorShape([])
                ),
                states=states
            )

            def condition(_time, _hypothesis): return tf.less(_time, total)

            def body(_time, _hypothesis):
                ytu, _states = self.decoder_inference(
                    # avoid using [index] in tflite
                    encoded=tf.gather_nd(encoded, tf.reshape(_time, shape=[1])),
                    predicted=_hypothesis.index,
                    states=_hypothesis.states
                )
                _predict = tf.argmax(ytu, axis=-1, output_type=tf.int32)  # => argmax []

                _equal = tf.equal(_predict, self.text_featurizer.blank)
                _index = tf.where(_equal, _hypothesis.index, _predict)
                _states = tf.where(_equal, _hypothesis.states, _states)
                _time = tf.where(_equal, _time + 1, _time)

                _prediction = _hypothesis.prediction.write(_time, _predict)
                _hypothesis = Hypothesis(index=_index, prediction=_prediction, states=_states)

                return _time, _hypothesis

            time, hypothesis = tf.while_loop(
                condition, body,
                loop_vars=[time, hypothesis], parallel_iterations=parallel_iterations, swap_memory=swap_memory
            )

            return Hypothesis(index=hypothesis.index, prediction=hypothesis.prediction.stack(), states=hypothesis.states)

