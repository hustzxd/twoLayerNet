from util.read_mat import ReadMat
from classifiers.neural_net import TwoLayerNet
import matplotlib.pyplot as plt

myReader = ReadMat('data/homework.mat')
X, y, X_val, y_val = myReader.getData()

input_size = 2
hidden_size = 1
output_size = 2

net = TwoLayerNet(input_size, hidden_size, output_size)

stats = net.train(X, y, X, y,
                  learning_rate=1e-1,
                  num_iters=1000)
print('Final training loss: {}'.format(stats['loss_history'][-1]))

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

plt.plot(stats['train_acc_history'])
plt.xlabel('iteration')
plt.ylabel('training accuracy')
plt.title('Training accuracy history')
plt.show()

plt.plot(stats['val_acc_history'])
plt.xlabel('iteration')
plt.ylabel('val accuracy')
plt.title('val accuracy history')
plt.show()



