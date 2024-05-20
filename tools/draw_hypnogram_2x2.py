import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec


def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='medium')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')


def main():
    convert_index = {0:4, 4:3, 1:2, 2:1, 3:0}
    matplotlib.rcParams['font.family'] = ['Ubuntu']
    matplotlib.rcParams['font.size'] = 12
    
    y = np.load('./source/02-6_true.npy', allow_pickle=True)
    y_pred = np.load('./source/02-6_pred.npy', allow_pickle=True)
    
    for i in range(y.__len__()):
        y[i] = convert_index[y[i]]
        y_pred[i] = convert_index[y_pred[i]]
    x = np.array(range(0, y.__len__()))/2/60
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    
    axs[0][0].plot(x, y, '#00008B')
    axs[0][0].set_yticks([0,1,2,3,4], ['N3','N2','N1','REM','Wake'])
    axs[0][0].set_xlabel('Time [h]')
    axs[0][0].set_ylabel('Sleep Stage')
    
    axs[1][0].plot(x, y_pred, '#008000')
    axs[1][0].set_yticks([0,1,2,3,4], ['N3','N2','N1','REM','Wake'])
    axs[1][0].set_xlabel('Time [h]')
    axs[1][0].set_ylabel('Sleep Stage')

    y = np.load('./source/07-5_true.npy', allow_pickle=True)
    y_pred = np.load('./source/07-5_pred.npy', allow_pickle=True)
    
    for i in range(y.__len__()):
        y[i] = convert_index[y[i]]
        y_pred[i] = convert_index[y_pred[i]]
    x = np.array(range(0, y.__len__()))/2/60
    
    axs[0][1].plot(x, y, '#00008B')
    axs[0][1].set_yticks([0,1,2,3,4], ['N3','N2','N1','REM','Wake'])
    axs[0][1].set_xlabel('Time [h]')
    axs[0][1].set_ylabel('Sleep Stage')
    
    axs[1][1].plot(x, y_pred, '#008000')
    axs[1][1].set_yticks([0,1,2,3,4], ['N3','N2','N1','REM','Wake'])
    axs[1][1].set_xlabel('Time [h]')
    axs[1][1].set_ylabel('Sleep Stage')
    
    grid = plt.GridSpec(2, 2)
    create_subtitle(fig, grid[0, ::], 'Hypnogram Scored by Human Expert')
    create_subtitle(fig, grid[1, ::], 'Hypongram Scored by SleePyCo')

    fig.tight_layout()
    fig.savefig('./hypnogram.png', bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    main()