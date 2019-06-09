import matplotlib.pyplot as plt
import matplotlib

# plot the loss history
matplotlib.interactive(True)


def plot_loss(stats):
    plt.plot(stats['loss_history'])
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.plot()
    plt.show()
