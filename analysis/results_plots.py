import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ogrit.core.base import get_base_dir, get_all_scenarios

scenarios = get_all_scenarios()

lanelet_acc = {'heckstrasse': [0.9259259259259259, 0.9230769230769231, 0.9287749287749287, 0.9430199430199431, 0.9430199430199431, 0.9458689458689459, 0.9487179487179487, 0.9601139601139601, 0.9515669515669516, 0.9572649572649573, 0.9971509971509972],
               'bendplatz': [0.7751479289940828, 0.8284023668639053, 0.8816568047337278, 0.8816568047337278, 0.9171597633136095, 0.9289940828402367, 0.9467455621301775, 0.9526627218934911, 0.9585798816568047, 0.9704142011834319, 0.9881656804733728],
               'frankenburg': [0.7023809523809523, 0.6904761904761905, 0.6904761904761905, 0.6845238095238095, 0.7083333333333334, 0.7023809523809523, 0.7440476190476191, 0.8333333333333334, 0.8869047619047619, 0.9523809523809523, 0.9702380952380952],
               'neuweiler': [0.8127490039840638, 0.7948207171314741, 0.7768924302788844, 0.7888446215139442, 0.7868525896414342, 0.8227091633466136, 0.8705179282868526, 0.9143426294820717, 0.9601593625498008, 0.9760956175298805, 0.9940239043824701]}

lanelet_sem = {'heckstrasse': [0.013998684185526933, 0.01424338615034695, 0.013747941191741588, 0.012390472155953016, 0.012390472155953021, 0.012094967443376138, 0.011790092995920206, 0.010460148006088785, 0.011475102022892925, 0.010811205675789332, 0.0028490028490028496],
               'bendplatz': [0.03220965704514525, 0.02908852308432471, 0.024921075824304247, 0.024921075824304247, 0.02126613279373553, 0.019815229365563895, 0.017323669933997633, 0.016383873496879833, 0.015373232019842106, 0.013072708604860005, 0.008343185274439521],
               'frankenburg': [0.03538005376298528, 0.03577364139046235, 0.03577364139046235, 0.03595997140191399, 0.035172561963882104, 0.03538005376298527, 0.03376927346216977, 0.028838689175249278, 0.024507692207505812, 0.016479250957285304, 0.013149561998772784],
               'neuweiler': [0.01742895819738689, 0.018041908672949226, 0.01860024799988408, 0.018233831851723836, 0.018296493115113152, 0.017062687981146858, 0.01499943211287082, 0.012503108227838839, 0.008738078397871425, 0.006824425278140986, 0.003443406699477138]}


opendrive_acc = {'heckstrasse': [0.8636363636363636, 0.8796791443850267, 0.893048128342246, 0.9251336898395722, 0.9197860962566845, 0.9331550802139037, 0.9224598930481284, 0.9545454545454546, 0.9652406417112299, 0.9759358288770054, 0.9919786096256684],
               'bendplatz': [0.6977777777777778, 0.7288888888888889, 0.8222222222222222, 0.8444444444444444, 0.8844444444444445, 0.8844444444444445, 0.9022222222222223, 0.9066666666666666, 0.9511111111111111, 0.9644444444444444, 0.9733333333333334],
               'frankenburg': [0.7743055555555556, 0.7673611111111112, 0.75, 0.78125, 0.8229166666666666, 0.8576388888888888, 0.8611111111111112, 0.8715277777777778, 0.9375, 0.9791666666666666, 0.9756944444444444]}

opendrive_sem = {'heckstrasse': [0.017768891320797175, 0.016845278200083687, 0.016002108507738953, 0.013626711050629453, 0.01406416103823163, 0.012931732121797808, 0.013847855668320183, 0.010785307968157682, 0.009484168450990522, 0.007934903853986201, 0.004618719246936513],
               'bendplatz': [0.030682995422756144, 0.02970163209073899, 0.02554520149116045, 0.024216105241892643, 0.02136026738895583, 0.021360267388955814, 0.019845078999435287, 0.019436506316151, 0.01440776784413328, 0.012372809695177825, 0.01076443290995935],
               'frankenburg': [0.024676051826979906, 0.024940209775433337, 0.02555993162216239, 0.024402150295094696, 0.022533353241886888, 0.02062561794274979, 0.020413731580620927, 0.019751691176171152, 0.014288436151850391, 0.008430760327857062, 0.009090101735590451]}


for scenario in scenarios:
    acc = pd.read_csv(get_base_dir() + f'/results/{scenario}_trained_trees_acc.csv')
    acc_sem = pd.read_csv(get_base_dir() + f'/results/{scenario}_trained_trees_acc_sem.csv')
    opendrive_acc[scenario] = acc.model_correct.to_numpy()
    opendrive_sem[scenario] = acc_sem.model_correct.to_numpy()

plt.style.use('ggplot')
fraction_observerd = np.linspace(0, 1, 11)
for scenario in scenarios:
    plt.figure()
    plt.title(scenario)
    plt.xlabel('fraction of trajectory observed')
    plt.ylabel('GRIT accuracy')

    accuracy = np.array(lanelet_acc[scenario])
    accuracy_sem = np.array(lanelet_sem[scenario])
    plt.plot(fraction_observerd, accuracy, label='lanelet2')
    plt.fill_between(fraction_observerd, (accuracy + accuracy_sem),
                     (accuracy - accuracy_sem), alpha=0.2)

    accuracy = np.array(opendrive_acc[scenario])
    accuracy_sem = np.array(opendrive_sem[scenario])
    plt.plot(fraction_observerd, accuracy, label='OpenDrive')
    plt.fill_between(fraction_observerd, (accuracy + accuracy_sem),
                     (accuracy - accuracy_sem), alpha=0.2)

    plt.legend()
    plt.ylim([0, 1])
    plt.show()
