import numpy as np


class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        :param input_size:
        :param hidden_size:
        :param output_size:
        :param std:

        W1 shape(I, H)
        b1 shape(H)
        W2 shape(H, O)
        b2 shape(O)

        """
        self.params = {'W1': std * np.random.randn(input_size, hidden_size), 'b1': np.zeros(hidden_size),
                       'W2': std * np.random.randn(hidden_size, output_size), 'b2': np.zeros(output_size)}

    def loss(self, train_batch, label_batch):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, I = train_batch.shape

        # forward
        fc_1 = np.dot(train_batch, W1) + b1  # (N,I) x (I, H)  = (N, H)
        relu_1 = np.maximum(0, fc_1)
        outputs = np.dot(relu_1, W2) + b2  # (N, H) x (H, O) = (N, O)

        # Softmax Loss
        outputs -= np.max(outputs)
        s_exp = np.exp(outputs)
        s_exp_sum = np.sum(s_exp, axis=1)
        correct_scores = outputs[np.arange(N), label_batch]
        loss = np.sum(-correct_scores + np.log(s_exp_sum)) / N

        grads = {}
        binary_scores = np.zeros(outputs.shape)
        binary_scores[np.arange(N), label_batch] = -1
        final_scores = binary_scores + (s_exp / s_exp_sum.reshape(-1, 1))
        grads['W2'] = relu_1.T.dot(final_scores / N)  # (H, O)
        grads['b2'] = np.sum(final_scores / N, axis=0)  # (1, O)

        # Score_1 is the layer that got ReLU'd
        dJdscore_1 = (final_scores / N).dot(W2.T)
        dJdscore_1[relu_1 == 0] = 0

        # Compute W1
        grads["W1"] = train_batch.T.dot(dJdscore_1)
        # grads["W1"] += reg * (W1)

        # Compute b1
        grads["b1"] = np.sum(dJdscore_1, axis=0)

        return loss, grads

    def train(self, train_data, label, val_data, val_label, learning_rate=1e-3, learning_rate_decay=0.95, num_iters=100,
              batch_size=200):
        """

        :param learning_rate_decay:
        :param train_data: shape(N, I)
        :param label: shape(N)
        :param val_data:
        :param label_val:
        :param learning_rate:
        :param num_iters:
        :param batch_size:
        :return:
        """
        num_train = train_data.shape[0]
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        iter_per_epoch = max(num_train / batch_size, 1)

        for it in range(num_iters):
            # Random select a mini batch from train data
            indices = np.random.choice(num_train, batch_size)
            train_batch = train_data[indices]
            label_batch = label[indices]

            loss, grads = self.loss(train_batch, label_batch)
            loss_history.append(loss)

            self.params['W1'] -= grads['W1'] * learning_rate
            self.params['b1'] -= grads['b1'] * learning_rate
            self.params['W2'] -= grads['W2'] * learning_rate
            self.params['b2'] -= grads['b2'] * learning_rate

            if it % 100 == 0:
                print('iteration {:d}/{:d}: loss {:.3f}'.format(it, num_iters, loss))
            if it % iter_per_epoch == 0:
                train_acc = (self.predict(train_batch) == label_batch).mean()
                val_acc = (self.predict(val_data) == val_label).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history}

    def predict(self, X):
        scores = np.dot(np.maximum(0, (np.dot(X, self.params['W1']) + self.params['b1'])), self.params['W2']) + \
                 self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        return y_pred
