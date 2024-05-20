
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main(fold, subject_id):
    convert_index = {0:4, 4:3, 1:2, 2:1, 3:0}
    matplotlib.rcParams['font.family'] = ['Ubuntu']
    matplotlib.rcParams['font.size'] = 12
    
    y = np.load('./source/{:02d}-{}_true.npy'.format(fold, subject_id), allow_pickle=True)      # y: one-dimensional array with ground truth label
    y_pred = np.load('./source/{:02d}-{}_pred.npy'.format(fold, subject_id), allow_pickle=True) # y_pred: one-dimensional array with predicted label
    
    for i in range(y.__len__()):
        y[i] = convert_index[y[i]]
        y_pred[i] = convert_index[y_pred[i]]
    x = np.array(range(0, y.__len__())) / 2 / 60
    
    fig = plt.gcf()
    fig.clf()
    plt.figure(figsize=(10, 6))
    
    ax_1 = plt.subplot(2, 1, 1)
    ax_1.plot(x, y, '#00008B')
    plt.title('Hypnogram Scored by Human Expert', fontweight="medium")
    plt.yticks([0,1,2,3,4],['N3','N2','N1','REM','Wake'])
    plt.xlabel('Time [h]')
    plt.ylabel('Sleep Stage')

    ax_2 = plt.subplot(2, 1, 2)
    ax_2.plot(x, y_pred, '#008000')
    plt.title('Hypnogram Scored by SleePyCo', fontweight="medium")
    plt.yticks([0, 1, 2, 3, 4], ['N3', 'N2', 'N1', 'REM', 'Wake'])
    plt.xlabel('Time [h]')
    plt.ylabel('Sleep Stage')

    plt.tight_layout()
    plt.savefig('./{:02d}-{}.png'.format(fold, subject_id), bbox_inches="tight")

    plt.show()


if __name__ == '__main__':
    main(fold=7, subject_id=5)
    # for fold in range(1, 11):
    #     for subject_id in range(1, num_list[fold - 1] + 1):
    #         main(fold, subject_id)