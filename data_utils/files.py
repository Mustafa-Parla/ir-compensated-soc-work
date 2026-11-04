"""Panasonic 18650PF (Kollmeyer) dosya listeleri ve rolleri.
Roller: NN, C1–C4 (karışık çevrimler) × eğitim sıcaklıkları; test: US06 / HWFET(a) / UDDS / LA92 × 5 sıcaklık (−20°C eğitimde yok)."""
DATA_FOLDERS = {'25degC': 'DriveCycles25deg', '10degC': 'DriveCycles10deg', '0degC': 'DriveCycles0deg',
                '-10degC': 'DriveCyclesN10deg', '-20degC': 'DriveCyclesN20deg'}
TEMPS = ['25degC', '10degC', '0degC', '-10degC', '-20degC']
TRAIN_TEMPS = ['25degC', '10degC', '0degC', '-10degC']
TRAIN_FILES = {
    '25degC':  ['03-21-17_16.27 25degC_NN_Pan18650PF.mat', '03-18-17_02.17 25degC_Cycle_1_Pan18650PF.mat',
                '03-19-17_03.25 25degC_Cycle_2_Pan18650PF.mat', '03-19-17_09.07 25degC_Cycle_3_Pan18650PF.mat'],
    '10degC':  ['03-27-17_09.06 10degC_NN_Pan18650PF.mat', '03-28-17_12.51 10degC_Cycle_1_Pan18650PF.mat',
                '03-28-17_18.18 10degC_Cycle_2_Pan18650PF.mat', '04-05-17_17.04 10degC_Cycle_3_Pan18650PF.mat'],
    '0degC':   ['06-01-17_10.36 0degC_NN_Pan18650PF.mat', '05-30-17_12.56 0degC_Cycle_1_Pan18650PF.mat',
                '05-30-17_20.16 0degC_Cycle_2_Pan18650PF.mat', '06-01-17_15.36 0degC_Cycle_3_Pan18650PF.mat'],
    '-10degC': ['06-14-17_13.12 n10degC_NN_Pan18650PF.mat', '06-10-17_11.25 n10degC_Cycle_1_Pan18650PF.mat',
                '06-10-17_18.35 n10degC_Cycle_2_Pan18650PF.mat', '06-11-17_01.39 n10degC_Cycle_3_Pan18650PF.mat'],
}
VAL_FILES = {
    '25degC': ['03-19-17_14.31 25degC_Cycle_4_Pan18650PF.mat'], '10degC': ['04-05-17_22.50 10degC_Cycle_4_Pan18650PF.mat'],
    '0degC': ['06-01-17_22.03 0degC_Cycle_4_Pan18650PF.mat'],   '-10degC': ['06-11-17_08.42 n10degC_Cycle_4_Pan18650PF.mat'],
}
TEST_FILES = [
    ('25degC', 'US06', '03-20-17_01.43 25degC_US06_Pan18650PF.mat'), ('25degC', 'HWFET', '03-20-17_05.56 25degC_HWFTa_Pan18650PF.mat'),
    ('25degC', 'UDDS', '03-21-17_00.29 25degC_UDDS_Pan18650PF.mat'), ('25degC', 'LA92', '03-21-17_09.38 25degC_LA92_Pan18650PF.mat'),
    ('10degC', 'US06', '03-27-17_09.06 10degC_US06_Pan18650PF.mat'), ('10degC', 'HWFET', '03-27-17_09.06 10degC_HWFET_Pan18650PF.mat'),
    ('10degC', 'UDDS', '03-27-17_09.06 10degC_UDDS_Pan18650PF.mat'), ('10degC', 'LA92', '03-27-17_09.06 10degC_LA92_Pan18650PF.mat'),
    ('0degC', 'US06', '06-02-17_04.58 0degC_US06_Pan18650PF.mat'), ('0degC', 'HWFET', '06-02-17_10.43 0degC_HWFET_Pan18650PF.mat'),
    ('0degC', 'UDDS', '06-02-17_17.14 0degC_UDDS_Pan18650PF.mat'), ('0degC', 'LA92', '06-01-17_10.36 0degC_LA92_Pan18650PF.mat'),
    ('-10degC', 'US06', '06-07-17_08.39 n10degC_US06_Pan18650PF.mat'), ('-10degC', 'HWFET', '06-07-17_08.39 n10degC_HWFET_Pan18650PF.mat'),
    ('-10degC', 'UDDS', '06-07-17_08.39 n10degC_UDDS_Pan18650PF.mat'), ('-10degC', 'LA92', '06-07-17_08.39 n10degC_LA92_Pan18650PF.mat'),
    ('-20degC', 'US06', '06-25-17_10.31 n20degC_US06_Pan18650PF.mat'), ('-20degC', 'HWFET', '06-23-17_23.35 n20degC_HWFET_Pan18650PF.mat'),
    ('-20degC', 'UDDS', '06-23-17_23.35 n20degC_UDDS_Pan18650PF.mat'), ('-20degC', 'LA92', '06-23-17_23.35 n20degC_LA92_Pan18650PF.mat'),
]

ROLE_FILES = {t: {**dict(zip(['NN', 'C1', 'C2', 'C3'], TRAIN_FILES[t])), 'C4': VAL_FILES[t][0]} for t in TRAIN_FILES}
