import matplotlib.pyplot as plt
from numpy import ndarray, sqrt, arctan
from texttable import Texttable


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


def info_table(regr):
    print()
    table = Texttable()
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.set_cols_dtype(['t', 'a'])
    table.set_cols_valign(['m', 'm'])
    list_of_values = [["Parameter", "Value"]]
    list_of_values += [['n_freq', regr.n_freq], ['q_freq', regr.q_freq]]
    list_of_values += ([[str(key), str(value)] for key, value in regr.params.items()])
    list_of_values += [['================', '============='],
                       ['fitted', str(regr._fitted)],
                       ['total iterations', regr._total_iterations],
                       ['rmse loss', regr.loss[-1]],
                       ['# frequencies', len(regr.frequencies)],
                       ['sample frequency', round(regr.Fs, 7)]]

    table.add_rows(list_of_values)
    print(table.draw())
    print()


def result_table(regr):
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['e', 'f', 'f', 'f', 'f', 'f'])
    table.set_cols_valign(['m', 'm', 'm', 'm', 'm', 'm'])

    list_of_values = [["Frequency", "FS", "A(cos)", "B(sin)", 'Intencity', 'Phase']]
    intercept = round(regr.weights[0][0], 5)
    formula = f'Y = {intercept}'
    for n in range(len(regr.frequencies)):
        freq = round(regr.frequencies[n], 5)
        intencity = round(sqrt(regr.weights[3 * n + 2] ** 2 + regr.weights[3 * n + 3] ** 2)[0], 5)
        phase = round(arctan(regr.weights[3 * n + 3] / regr.weights[3 * n + 2])[0], 5)
        fs = round(regr.weights[3 * n + 1][0], 5)
        freq_fs = round(freq * fs, 5)
        A, B = round(regr.weights[3 * n + 2][0], 5), round([3 * n + 3][0], 5)
        list_of_values += [[regr.frequencies[n], fs, A, B, intencity, phase]]
        if phase > 0:
            formula += f" + {intencity}∙cos(2∙pi∙{freq_fs}∙X - {phase})"
        else:
            formula += f" + {intencity}∙cos(2∙pi∙{freq_fs}∙X + {-phase})"
    list_of_values += [['intercept', regr.weights[0], '', '', '', '']]

    table.add_rows(list_of_values)
    print(table.draw())
    try:
        print('\nTotal Formula\n===============')
        print("Y = A₀ + Σ[ Aᵢ∙cos(2π∙fᵢ∙fsᵢ∙X - φᵢ) ]")
        print(formula)
        print()
    except:
        pass