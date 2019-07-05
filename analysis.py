import matplotlib.pyplot as plt
import matplotlib

# plot the loss history
matplotlib.interactive(True)


def plot_loss(stats):
    plt.figure()
    plt.plot(stats['loss_history'])
    plt.plot(stats['loss_val_history'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss history')
    plt.legend(['Training loss', 'Validation loss'])
    plt.plot()
    plt.show()

def plot_f1(stats):
    plt.figure()
    plt.plot(stats['f1_history'])
    plt.plot(stats['f1_val_history'])
    plt.xlabel('Iteration')
    plt.ylabel('F1 score')
    plt.title('F1 score history')
    plt.legend(['Training F1', 'Validation F1'])
    plt.plot()
    plt.show()