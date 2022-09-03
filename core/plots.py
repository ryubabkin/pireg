import matplotlib.pyplot as plt
from numpy import ndarray


def plot_loss(
        loss: list,
        save_to: str = None,
        log: bool = False
):
    plt.figure(figsize=(10, 7))
    plt.plot(loss, c='blue')
    plt.axhline(0, c='lightgray')
    if log:
        plt.yscale('log')
        plt.ylabel('RMSE loss [log scale]', fontsize=15)
    else:
        plt.ylabel('RMSE loss', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Epochs', fontsize=15)
    plt.title(f'Loss {round(loss[-1], 5)} for epoch {int(len(loss))}', fontsize=15)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    plt.show()


def plot_spectrum(
        spectrum: ndarray,
        top_spectrum: ndarray,
        save_to: str = None,
        log: bool = False
):
    plt.figure(figsize=(10, 7))
    plt.plot(spectrum[0, :],
             spectrum[1, :], c='gray')
    plt.plot(spectrum[0, spectrum[3] == 1],
             spectrum[1, spectrum[3] == 1],
             'x', c='b')
    plt.plot(top_spectrum[0, :],
             top_spectrum[1, :],
             'x', c='r', markersize=10)
    if log:
        plt.xscale('log')
        plt.xlabel('Frequency (log scale)', fontsize=15)
    else:
        plt.xlabel('Frequency', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('Intensity [arb.units]', fontsize=15)
    plt.title('Signal spectrum', fontsize=15)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    plt.show()
