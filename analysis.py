import matplotlib.pyplot as plt
import matplotlib

# plot the loss history
matplotlib.interactive(True)


def plot_loss(stats):
    plt.plot(stats['loss_history'])
    plt.plot(stats['loss_val_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss history')
    plt.legend(['Training loss', 'Validation loss'])
    plt.plot()
    plt.show()
