import resonance
import resonance.pipe
import resonance.cross
import scipy.signal as sp_sig
import numpy as np
import time
import threading

WorkState = "None"
EpochsForTraining = []

# селективная фильтрация каналов + корректировка ЭЭГ на референт
def FiltrationAndRefCorrection(allData):
    np_eye = np.eye(allData.SI.channels)  # единичная матрица - основа пространственной фильтрации

    sptl_filtr0 = np_eye[...,:-1]  # 2 последний канал - эвентный канал, убираем его из списка каналов (убираем из единичной матрицы крайний столбец)
    woEvtChnlData = resonance.pipe.spatial(allData, sptl_filtr0)

    bandstop_frequency = 50  # применяем режектор 50Гц для всех данных (эвентный канал выкинули)
    w0 = [(bandstop_frequency - 0.5) / (allData.SI.samplingRate / 2),(bandstop_frequency + 0.5) / (allData.SI.samplingRate / 2)]#50+-0.5Гц
    bandstop_filter = sp_sig.butter(4, w0, btype='bandstop')
    woEvtChnldata_Notchfiltered = resonance.pipe.filter(woEvtChnlData, bandstop_filter)
    #resonance.createOutput(woEvtChnldata_Notchfiltered, 'AllData_Notch_Filtered')

    np_eye2 = np.eye(allData.SI.channels - 1)  # единичная матрица для всех каналов минус эвентный канал

    sptl_filtr = np_eye2[...,:-1]  # оставляем только ЭЭГ, убираем последний канал - это ЭМГ (убираем из единичной матрицы крайний столбец)
    eegData_Notchfiltered = resonance.pipe.spatial(woEvtChnldata_Notchfiltered, sptl_filtr)
    sptl_filtr2 = np_eye2[..., -1:]  # оставляем последний столбец - для отдельной фильтрации ЭМГ канала
    emgData_Notchfiltered = resonance.pipe.spatial(woEvtChnldata_Notchfiltered, sptl_filtr2)

    cut_off_frequency = 35
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / allData.SI.samplingRate * 2, btype='low')
    eeg_NotchLPfiltered = resonance.pipe.filter(eegData_Notchfiltered, low_pass_filter)
    #resonance.createOutput(eeg_NotchLPfiltered, 'EEG_Notch_LP_Filtered')

    highpass_frequency = 0.1
    highpass_filter = sp_sig.butter(4, highpass_frequency, 'hp', fs=allData.SI.samplingRate)
    eeg_NotchLPHPfiltered = resonance.pipe.filter(eeg_NotchLPfiltered, highpass_filter)
    #resonance.createOutput(eeg_NotchLPHPfiltered, 'EEG_Notch_LP_HPFiltered')

    # уже фильтрованную ЭЭГ корректируем на референт, в качестве которого возьмём 1й канал
    np_eye3 = np.eye(allData.SI.channels - 2)  # единичная матрица для всех каналов минус эвентный канал и минус ЭМГ
    np_eye3[0, ...] = -1  # готовим матрицу, чтобы 1й канал вычесть из остальных (нулевая строка = -1)
    sptl_filtr4 = np_eye3[...,1:]  # первый канал - это референт, убираем его из списка каналов (Убираем первый столбец матрицы)
    eeg_Filtered_Referenced = resonance.pipe.spatial(eeg_NotchLPHPfiltered, sptl_filtr4)
    #resonance.createOutput(eeg_Filtered_Referenced, 'EEG_Filtered_Referenced')

    # для ЭМГ применим только фильтр высоких частот
    highpass_frequency = 5
    highpass_filter = sp_sig.butter(4, highpass_frequency, 'hp', fs=allData.SI.samplingRate)
    emg_HPfiltered = resonance.pipe.filter(emgData_Notchfiltered, highpass_filter)
    #resonance.createOutput(emg_HPfiltered, 'EMG_Notch_HPFiltered')
    return eeg_Filtered_Referenced, emg_HPfiltered

# заглушка для отладки с искусственным сигналом
def StubForDebugging(allData):
    eeg_Filtered_Referenced = allData
    emg_HPfiltered = allData
    return eeg_Filtered_Referenced, emg_HPfiltered

def worker():
    """thread worker function"""
    time.sleep(10)
    with open("d:/Projects/BCI_EyeLines_Online_2020/rpe/worker.txt", "at") as f:
        f.write("worker\n")
        #np.savetxt(f, 0)
    return

#меняем рабочий статус системы согласно входящим командам
def UpdateWorkState(evt):
    global WorkState
    if (evt == "1"):
        WorkState = 'NeedEpochingForTraining'
    elif (evt == "2"):
        WorkState = 'NeedTraining'
        task = threading.Thread(target=worker)
        task.start()
    return True

#просмотр максимумов шинкованных микроокон и выбор кандидатов на взятие окна для нецелевого класса
class GetEventsForUntargetedEpochs:
    def __init__(self, threshold, delay):
        self._threshold = threshold
        self._delay = delay
        self._last_event = -np.Infinity

    def __call__(self, evt):
        if WorkState == "NeedEpochingForTraining":
            #смотрим амплитуды по порогу
            if float(abs(evt) < self._threshold):
                current_ts = evt.timestamps[-1]
                if self._last_event < current_ts - self._delay * 1e9:  # minimal delay between events = 2 seconds
                    self._last_event = current_ts #берём подходящий по амплитуде эвент, только если он по времени больше заданного интервала
                    return True
        return False

class CheckEvtsForAmplitudeThreshold:
    def __init__(self, threshold):
        self._threshold = threshold

    def __call__(self, evt):
        if WorkState == "NeedEpochingForTraining":
            if float(abs(evt) < self._threshold):
                return True
        return False

#просмотр максимумов шинкованных микроокон и выбор кандидатов на взятие окна для целевого класса
class GetEventsForTargetedEpochs:
    def __init__(self, threshold, delay):
        self._threshold = threshold
        self._delay = delay
        self._last_event = -np.Infinity

    def __call__(self, evt):
        if WorkState == "NeedEpochingForTraining":
            #смотрим амплитуды по порогу
            if float(abs(evt) > self._threshold):
                current_ts = evt.timestamps[-1]
                if self._last_event < current_ts - self._delay * 1e9:  # minimal delay between events = 2 seconds
                    self._last_event = current_ts #берём подходящий по амплитуде эвент, только если он по времени больше заданного интервала
                    return True
        return False

def makeEvent(block):
    return abs(np.max(block))

def makeEvtAndAddUntargetEpochToList(block):
    if WorkState == "NeedEpochingForTraining":
        '''
        with open("d:/Projects/BCI_EyeLines_Online_2020/rpe/epoching.txt", "at") as f:
            f.write("UntargetEpoch\n")
            np.savetxt(f, block)
        '''
        global EpochsForTraining
        for item in block:
            EpochsForTraining.append(item)
        EpochsForTrainingArr = np.array(EpochsForTraining)
    return abs(np.max(block))

def makeEvtAndAddTargetEpochToList(block):
    if WorkState == "NeedEpochingForTraining":
        '''
        with open("d:/Projects/BCI_EyeLines_Online_2020/rpe/epoching.txt", "at") as f:
            f.write("TargetEpoch\n")
            np.savetxt(f, block)
        '''
        global EpochsForTraining
        for item in block:
            EpochsForTraining.append(item)
        EpochsForTrainingArr = np.array(EpochsForTraining)
    return abs(np.max(block))

def online_processing():
    alldata = resonance.input(0)
    #eeg_Filtered_Referenced, emg_Notch_HPFiltered = StubForDebugging(alldata)
    eeg_Filtered_Referenced, emg_Notch_HPFiltered = FiltrationAndRefCorrection(alldata)# селективная фильтрация каналов + корректировка ЭЭГ на референт
    resonance.createOutput(eeg_Filtered_Referenced, 'EEG_Filtered_Referenced')
    resonance.createOutput(emg_Notch_HPFiltered, 'EMG_Notch_HPFiltered')

    #просматриваем эвенты и меняем рабочий статус системы согласно входящим командам
    events = resonance.input(1)
    cmd_input = resonance.pipe.filter_event(events, UpdateWorkState)
    resonance.createOutput(cmd_input, 'EvtWorkState')#входящие команды эхом отправляем в эвентный поток для контроля

    window_size = 10
    window_shift = -10
    baseline_begin_offset = 0
    baseline_end_offset = 10
    thresholdForEEGInVolts = 0.5#0.0005  # пороговая амплитуда помехи для оценки кандидата на нецелевую эпоху
    thresholdForEMGInVolts = 0.05  # пороговая амплитуда ЭМГ для оценки кандидата на целевую эпоху
    intervalEvtToEvtInSec = 1  # интервал между кандидатами на целевую/нецелевую эпоху, сек

    #Формируем эпохи для нецелевого класса
    #просмотр шинкованной ЭЭГ на предмет кандидатов на формирование нецелевых эпох
    eeg_windows = resonance.pipe.windowizer(eeg_Filtered_Referenced, 10, 10)#шинкуем ЭЭГ на мелкие окна
    eeg_as_events = resonance.pipe.transform_to_event(eeg_windows, makeEvent)
    eeg_windowized = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, eeg_as_events, window_size, window_shift)
    EvtsMaxAmplitudes = resonance.pipe.transform_to_event(eeg_windowized, makeEvent)  # считаем максимумы амплитуд во взятых окнах
    evt = GetEventsForUntargetedEpochs(thresholdForEEGInVolts, intervalEvtToEvtInSec)#экземпляр получит значение, только в режиме NeedEpoching
    cndtnlEvtForUntargetEpochs = resonance.pipe.filter_event(EvtsMaxAmplitudes, evt)#оставляем эвенты только тех окон, которые проходят по порогу амплитуд и интервалов
    #resonance.createOutput(cndtnlEvtForUntargetEpochs, 'EvtCndtnlForUntargetEpochs')
    #берём окна для нецелевых эпох по отфильтрованным событиям (просмотр максимумов амплитуд (отбрасывание артефактных окон) по всему окну, интервалов между окнами)
    eeg_windowized_ = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, cndtnlEvtForUntargetEpochs, window_size, window_shift)
    baselinedEpoch = resonance.pipe.baseline(eeg_windowized_, slice(baseline_begin_offset, baseline_end_offset))
    #поскольку визуализации объекта окон пока нет, то преобразуем к эвенту чтобы посмотреть выхлоп во вьювере
    exhaustEvt = resonance.pipe.transform_to_event(baselinedEpoch, makeEvtAndAddUntargetEpochToList)
    resonance.createOutput(exhaustEvt, 'UntargetEpochEvt')
    #та же история взятия эпох для подачи обученному классификатору, только тогда intervalEvtToEvtInSec=0

    # Формируем целевые эпохи на основе порога амплитуды ЭМГ, интервалов между эвентами и безартефактности взятого окна ЭЭГ
    emg_windows = resonance.pipe.windowizer(emg_Notch_HPFiltered, 10, 10)#шинкуем ЭМГ на мелкие окна
    emg_as_events = resonance.pipe.transform_to_event(emg_windows, makeEvent)
    evt = GetEventsForTargetedEpochs(thresholdForEMGInVolts, intervalEvtToEvtInSec)#экземпляр получит значение, только в режиме NeedEpoching
    cndtnlEMGEvtForTargetEpochs = resonance.pipe.filter_event(emg_as_events, evt)#фильтранули событие по амплитуде ЭМГ
    #resonance.createOutput(cndtnlEvtForTargetEpochs, 'EvtCndtnlForTargetEpochs')
    eeg_wndwzd = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, cndtnlEMGEvtForTargetEpochs, window_size, window_shift)
    EvtsMaxAmplitudes = resonance.pipe.transform_to_event(eeg_wndwzd, makeEvent)  # считаем максимумы амплитуд во взятых окнах
    evt = CheckEvtsForAmplitudeThreshold(thresholdForEEGInVolts)#экземпляр получит значение, только в режиме NeedEpoching
    cndtnlEvtForTargetEpochs = resonance.pipe.filter_event(EvtsMaxAmplitudes, evt)#оставляем эвенты только тех окон, которые проходят по порогу амплитуд и интервалов
    #берём окна для целевых эпох по отфильтрованным событиям (просмотр максимумов амплитуд (отбрасывание артефактных окон) по всему окну)
    eeg_wndwzd_ = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, cndtnlEvtForTargetEpochs, window_size, window_shift)
    baselinedEpoch_ = resonance.pipe.baseline(eeg_wndwzd_, slice(baseline_begin_offset, baseline_end_offset))
    #поскольку визуализации объекта окон пока нет, то преобразуем к эвенту чтобы посмотреть выхлоп во вьювере
    exhaustEvt = resonance.pipe.transform_to_event(baselinedEpoch_, makeEvtAndAddTargetEpochToList)
    resonance.createOutput(exhaustEvt, 'TargetEpochEvt')