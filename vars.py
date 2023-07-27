window_size = 250
window_shift = -250  #
baseline_begin_offset = 0
baseline_end_offset = 50
thresholdForEEGInVolts = 0.005  # 0.0005  # пороговая амплитуда помехи для оценки кандидата на нецелевую эпоху 500мкВ
thresholdForEMGInVolts = 0.037  # 0.05  # пороговая амплитуда ЭМГ для оценки кандидата на целевую эпоху
#intervalEvtToEvtInSec = 5  # интервал между кандидатами на целевую/нецелевую эпоху, сек
intrvlEvtToEvtInSecForUntargetedEpochs = 2#5
intrvlEvtToEvtInSecForTargetedEpochs = 2#2

alldata_bandstop_frequency = 50

eeg_highpass_frequency = 0.1
eeg_cut_off_frequency = 25

emg_highpass_frequency = 5

proportion_of_the_TestDataset = 0.2
resample_data_to = 369

path_to_current_model = ""

model_predictions_threshold = 0.30 #порог классификатора, выше которого отправляется событие в стимулпрогу и считается, что классификатор распознал намерение (или False Alarm)
