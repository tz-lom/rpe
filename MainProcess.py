import resonance
import resonance.pipe
import resonance.cross
import scipy.signal as sp_sig
import numpy as np
import time
import threading
from enum import Enum
from mne.filter import resample
from sklearn.model_selection import train_test_split
from train_clf import train_model
import os
from utils import read_params, write_params
import glob
from keras.models import load_model
import vars

class WS(Enum):
    None_ = 0
    NeedEpochingForTraining = 1
    NeedPrepareEpochs = 2
    NeedTraining = 3
    PrepareForClassifierApply = 4
    TooFewEpochsForTraining = 5
    ClassifierApply = 6
    SettingsChanged = 7
    Reserved = 8

Trained_model = None

WorkState = WS.None_

TSFirstDataValue = 0;

X_Epochs = []
Y_Epochs = []

X_Tr = []
Y_Tr = []

X_Tst = []
Y_Tst = []

ClassifierPrepared = False

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

def worker():
    """thread worker function"""
    global X_Tr, Y_Tr
    logFileName = os.path.join(os.path.dirname(__file__), "workerStart.txt")
    with open(logFileName, "at") as f:
        f.write("Y_Tr:\n")
        np.savetxt(f, Y_Tr, fmt='%.1d', delimiter=',', newline='')

    global Trained_model

    Trained_model, val_auc = train_model(X_Tr, Y_Tr)
    val_auc = np.array(val_auc)
    logFileName = os.path.join(os.path.dirname(__file__), "train_model.txt")
    with open(logFileName, "at") as f:
        f.write("val_auc:\n")
        np.savetxt(f, [val_auc], fmt='%.18g')

    return

def loadModelThread():
    global Trained_model
    params = read_params('config.json')
    path_to_models_dir = params['path_to_models_dir']
    path_to_model = os.path.dirname(__file__) + path_to_models_dir + '/**/*.hdf5'
    filename_list = glob.glob(path_to_model)
    if len(filename_list) > 0:
        filename_ = filename_list[-1]
        Trained_model = load_model(filename_)

    global ClassifierPrepared
    if Trained_model is None:
        ClassifierPrepared = False
    else:
        ClassifierPrepared = True

    return

#меняем рабочий статус системы согласно входящим командам
def UpdateWorkState(evt):
    global WorkState
    if (evt == "1"):
        WorkState = WS.NeedEpochingForTraining
    elif (evt == "2"):
        WorkState = WS.NeedPrepareEpochs
        global X_Tr, Y_Tr, X_Tst, Y_Tst
        #почему нет объявления global X_Epochs global Y_Epochs ? Нихрена не понятно
        X_Tr, Y_Tr, X_Tst, Y_Tst = EpochsPrepare(X_Epochs, Y_Epochs)
        #для контроля эпох в матлабе: открываем там ЭЭГ и текстовый файл с метками времен
        logFileName = os.path.join(os.path.dirname(__file__), "AllEpochsTS.txt")
        with open(logFileName, "wt") as f:
            ts0 = []
            ts = []
            for item in X_Epochs:
                ts0.append(item.timestamps[0])
                ts.append(item.timestamps[-1]) #- TSFirstDataValue)/1000000000)
            ts0 = np.array(ts0)
            ts = np.array(ts)
            f.write("latency type position\n")
            array_1d = np.ones(len(Y_Epochs))
            np.savetxt(f, np.column_stack([ts0, Y_Epochs, array_1d]), fmt='%.18g')
            Y_Epochs_ = []
            # for item in Y_Epochs:
            #     Y_Epochs_.append(item + 10)
            array2_1d = np.full(len(Y_Epochs),2)
            np.savetxt(f, np.column_stack([ts, Y_Epochs, array2_1d]), fmt='%.18g')
        #записать в лог сколько подготовлено эпох и каковы их классы
        logFileName = os.path.join(os.path.dirname(__file__), "PrepareEpochs.txt")
        with open(logFileName, "wt") as f:
            f.write("PrepareEpochs\n")
            f.write("TrainingEpochsCnt: ")
            np.savetxt(f, [len(X_Tr)], fmt='%.1d')
            f.write("Y_TrainingEpochs: \n")
            np.savetxt(f, Y_Tr, fmt='%.1d', delimiter=',', newline='')

            f.write("\nTstEpochsCnt: ")
            np.savetxt(f, [len(X_Tst)], fmt='%.1d')
            f.write("Y_TstEpochs: \n")
            np.savetxt(f, Y_Tst, fmt='%.1d', delimiter=',', newline='')
        #лучше потом в управляющую прогу заслать значения
    elif (evt == "3"):
        #Надо проверять, если ли возможность обучать классификатор
        if len(Y_Tr) > 10: #!! вынести в настройки - минимальное кол-во обучающих эпох
            WorkState = WS.NeedTraining
            '''
            with open("d:/Projects/BCI_EyeLines_Online_2020/rpe/workerStart.txt", "at") as f:
                f.write("Y_Tr:\n")
                np.savetxt(f, Y_Tr, fmt='%.1d', delimiter=',', newline='')
            Trained_model, val_auc = train_model(X_Tr, Y_Tr)
            val_auc = np.array(val_auc)
            with open("d:/Projects/BCI_EyeLines_Online_2020/rpe/train_model.txt", "at") as f:
                f.write("val_auc:\n")
                np.savetxt(f, [val_auc], fmt='%.18g')
            '''
            task = threading.Thread(target=worker)
            task.start()
        else:
            WorkState = WS.TooFewEpochsForTraining
            return False
    elif (evt == "4"):
        WorkState = WS.PrepareForClassifierApply

        task = threading.Thread(target = loadModelThread)#загрузка обученной модели из файла
        task.start()

    elif (evt == "6"):
        WorkState = WS.ClassifierApply

    elif (evt == "7"):
        '''считывание настроек из файла, в который они сохраняются из полей контролов'''
        tmpVar = WorkState
        #WorkState = WS.SettingsChanged
        params = read_params('bin\SStim.json')

        vars.window_size = params['window_size']
        vars.window_shift = params['window_shift']
        vars.baseline_begin_offset = params['baseline_begin_offset']
        vars.baseline_end_offset = params['baseline_end_offset']
        vars.thresholdForEEGInVolts = params['thresholdForEEGInVolts']  # пороговая амплитуда помехи для оценки кандидата на нецелевую эпоху 500мкВ
        vars.thresholdForEMGInVolts = params['thresholdForEMGInVolts']  # пороговая амплитуда ЭМГ для оценки кандидата на целевую эпоху
        vars.intervalEvtToEvtInSec = params['intervalEvtToEvtInSec']  # интервал между кандидатами на целевую/нецелевую эпоху, сек
        #WorkState = tmpVar

    elif (evt == "8"):
        params = read_params('bin\SStim.json')
        param1 = params['intervalEvtToEvtInSec']
        WorkState = param1

    return True

#просмотр максимумов шинкованных микроокон и выбор кандидатов на взятие окна для нецелевого класса
class GetEventsForUntargetedEpochs:
    def __init__(self, threshold, delay):
        self._threshold = threshold
        self._delay = delay
        self._last_event = -np.Infinity

    def __call__(self, evt):
        if WorkState == WS.NeedEpochingForTraining:
            #смотрим амплитуды по порогу
            if float(abs(evt) < self._threshold):
                current_ts = evt.timestamps[-1]
                if self._last_event < current_ts - self._delay * 1e9:  # minimal delay between events = T seconds
                    self._last_event = current_ts #берём подходящий по амплитуде эвент, только если он по времени больше заданного интервала
                    return True
        elif WorkState == WS.ClassifierApply:
            if float(abs(evt) < self._threshold):
                return True
        return False

class GetFirstTSData:
    def __init__(self, needTS):
        self._needTS = needTS
    def __call__(self, evt):
        if self._needTS == True:
            global TSFirstDataValue
            TSFirstDataValue = evt.timestamps[0]
            self._needTS = False
            return True
        return False


class CheckEvtsForAmplitudeThreshold:
    def __init__(self, threshold):
        self._threshold = threshold

    def __call__(self, evt):
        if WorkState == WS.NeedEpochingForTraining:
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
        if WorkState == WS.NeedEpochingForTraining:
            #смотрим амплитуды по порогу
            if float(abs(evt) > self._threshold):
                current_ts = evt.timestamps[-1]
                if self._last_event < current_ts - self._delay * 1e9:  # minimal delay between events = 2 seconds
                    self._last_event = current_ts #берём подходящий по амплитуде эвент, только если он по времени больше заданного интервала
                    return True
        return False

def makeEvent(block):
    return abs(np.max(block))

def _resample(X, source_sample_rate, resample_to):
    '''
    Resample OVER 0-st axis
    :param X: eeg Time x Channels
    :param resample_to:
    :return:
    '''
    downsample_factor = source_sample_rate / resample_to
    return resample(X, up=1., down=downsample_factor, npad='auto', axis=0)

def EpochsPrepare(Xepochs, Yepochs):
    resample_to = 369
    sourceSR = 500  # переделать потом! сделать класс и передавать SR в него в init
    resampledEpochs = []
    for item in Xepochs:
        dataResampled = _resample(np.array(item), sourceSR, resample_to)
        resampledEpochs.append(dataResampled)
    X = np.array(resampledEpochs)
    X = X.transpose([0, 2, 1])
    x_tr_val_ind, x_tst_ind, y_tr_val, y_tst = train_test_split(range(X.shape[0]), Yepochs, test_size=0.2, stratify=Yepochs)
    x_tr_val = X[x_tr_val_ind, ...]
    x_tst = X[x_tst_ind, ...]
    return x_tr_val, y_tr_val, x_tst, y_tst

def makeEvtAndAddUntargetEpochToList(block):
    if WorkState == WS.NeedEpochingForTraining:
        '''
        with open("d:/Projects/BCI_EyeLines_Online_2020/rpe/epoching.txt", "at") as f:
            f.write("UntargetEpoch\n")
            np.savetxt(f, block)
        '''
        global X_Epochs
        global Y_Epochs
        _delay = 2  # чтобы не было коллизий с крайней эпохой; выдерживается интервал между нецелевыми и целевыми эпохами !!! вынести в глобальную переменную, что ли?

        if (len(X_Epochs) > 0) and (len(Y_Epochs) > 0):
            predEpoch = X_Epochs[-1]
            # если время начала новой эпохи минус время конца эпохи в списке больше заданного интервала, добавляем данную эпоху в список
            bgnDTNewEpoch = block.timestamps[0]#время начала новой эпохи
            endDTOldEpoch = predEpoch.timestamps[-1]#время конца крайней в списке эпохи
            if bgnDTNewEpoch - endDTOldEpoch > _delay * 1e9:
                # X_Epochs.pop(-1)# по факту удаляется предыдущая эпоха! Так было раньше 07.12.2021
                # Y_Epochs.pop(-1)
                X_Epochs.append(block)
                Y_Epochs.append(0)
        else:
            X_Epochs.append(block)
            Y_Epochs.append(0)

        return len(np.where(np.array(Y_Epochs) == 0)[0])
    elif WorkState == WS.NeedPrepareEpochs:
        return 123
    elif WorkState == WS.ClassifierApply:
        global ClassifierPrepared
        if ClassifierPrepared:
            resample_to = 369
            sourceSR = 500  # переделать потом! сделать класс и передавать SR в него в init
            dataResampled = _resample(np.array(block), sourceSR, resample_to)
            dataResampled = dataResampled.transpose([1, 0])#исходно строки - отсчёта, столбцы - каналы. Надо наоборот
            dataResampled = dataResampled[np.newaxis, np.newaxis, :, :]
            global Trained_model
            predictions = Trained_model.predict(dataResampled)[:, 1]
            return predictions[0]
            '''
            if predictions > 0.6:
                return 555
            else:
                return 0   
            '''
        else:
            return 345
    return abs(np.max(block))

def makeEvtAndAddTargetEpochToList(block):
    if WorkState == WS.NeedEpochingForTraining:
        '''
        with open("d:/Projects/BCI_EyeLines_Online_2020/rpe/epoching.txt", "at") as f:
            f.write("TargetEpoch\n")
            np.savetxt(f, block)
        '''
        global X_Epochs
        global Y_Epochs

        if (len(X_Epochs) > 0) and (len(Y_Epochs) > 0):
            predEpoch = X_Epochs[-1]
            _delay = 1 #!!! вынести в глобальную переменную, что ли?
            # если время начала новой эпохи минус время конца эпохи в списке меньше заданного интервала, удаляем данную эпоху из списка
            endDTNewEpoch = block.timestamps[-1]#время конца новой эпохи
            endDTOldEpoch = predEpoch.timestamps[-1]#время конца крайней в списке эпохи
            if endDTNewEpoch - endDTOldEpoch < _delay * 1e9:
                X_Epochs.pop(-1)
                Y_Epochs.pop(-1)

        X_Epochs.append(block)
        Y_Epochs.append(1)
    return len(np.where(np.array(Y_Epochs) == 1)[0])

def online_processing():
    alldata = resonance.input(0)
    eeg_Filtered_Referenced, emg_Notch_HPFiltered = FiltrationAndRefCorrection(alldata)# селективная фильтрация каналов + корректировка ЭЭГ на референт
    resonance.createOutput(eeg_Filtered_Referenced, 'EEG_Filtered_Referenced')
    resonance.createOutput(emg_Notch_HPFiltered, 'EMG_Notch_HPFiltered')

    #просматриваем эвенты и меняем рабочий статус системы согласно входящим командам
    events = resonance.input(1)
    cmd_input = resonance.pipe.filter_event(events, UpdateWorkState)
    resonance.createOutput(cmd_input, 'EvtWorkState')#входящие команды эхом отправляем в эвентный поток для контроля

    #Формируем эпохи для нецелевого класса
    #просмотр шинкованной ЭЭГ на предмет кандидатов на формирование нецелевых эпох
    eeg_windows = resonance.pipe.windowizer(eeg_Filtered_Referenced, 50, 50)#шинкуем ЭЭГ на мелкие окна
    eeg_as_events = resonance.pipe.transform_to_event(eeg_windows, makeEvent)

    #eeg_as_events_ = resonance.pipe.transform_to_event(eeg_windows, GetFirstTSData(True))

    eeg_windowized = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, eeg_as_events, vars.window_size, vars.window_shift)
    EvtsMaxAmplitudes = resonance.pipe.transform_to_event(eeg_windowized, makeEvent)  # считаем максимумы амплитуд во взятых окнах
    evt = GetEventsForUntargetedEpochs(vars.thresholdForEEGInVolts, vars.intervalEvtToEvtInSec)#экземпляр получит значение, только в режиме NeedEpoching
    cndtnlEvtForUntargetEpochs = resonance.pipe.filter_event(EvtsMaxAmplitudes, evt)#оставляем эвенты только тех окон, которые проходят по порогу амплитуд и интервалов
    #resonance.createOutput(cndtnlEvtForUntargetEpochs, 'EvtCndtnlForUntargetEpochs')
    #берём окна для нецелевых эпох по отфильтрованным событиям (просмотр максимумов амплитуд (отбрасывание артефактных окон) по всему окну, интервалов между окнами)
    eeg_windowized_ = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, cndtnlEvtForUntargetEpochs, vars.window_size, vars.window_shift)
    baselinedEpoch = resonance.pipe.baseline(eeg_windowized_, slice(vars.baseline_begin_offset, vars.baseline_end_offset))
    #поскольку визуализации объекта окон пока нет, то преобразуем к эвенту чтобы посмотреть выхлоп во вьювере
    exhaustEvt = resonance.pipe.transform_to_event(baselinedEpoch, makeEvtAndAddUntargetEpochToList)
    resonance.createOutput(exhaustEvt, 'UntargetEpochEvt')
    #та же история взятия эпох для подачи обученному классификатору, только тогда intervalEvtToEvtInSec=0

    # Формируем целевые эпохи на основе порога амплитуды ЭМГ, интервалов между эвентами и безартефактности взятого окна ЭЭГ
    emg_windows = resonance.pipe.windowizer(emg_Notch_HPFiltered, 50, 50)#шинкуем ЭМГ на мелкие окна
    emg_as_events = resonance.pipe.transform_to_event(emg_windows, makeEvent)
    #intervalEvtToEvtInSec = 2.8
    evt1 = GetEventsForTargetedEpochs(vars.thresholdForEMGInVolts, vars.intervalEvtToEvtInSec)#экземпляр получит значение, только в режиме NeedEpoching
    cndtnlEMGEvtForTargetEpochs = resonance.pipe.filter_event(emg_as_events, evt1)#фильтранули событие по амплитуде ЭМГ и интервалам между событиями
    #resonance.createOutput(cndtnlEvtForTargetEpochs, 'EvtCndtnlForTargetEpochs')
    eeg_wndwzd = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, cndtnlEMGEvtForTargetEpochs, vars.window_size, vars.window_shift)
    #baselinedEpoch_ = resonance.pipe.baseline(eeg_wndwzd, slice(baseline_begin_offset, baseline_end_offset))

    EvtsMaxAmplitudes = resonance.pipe.transform_to_event(eeg_wndwzd, makeEvent)  # считаем максимумы амплитуд во взятых окнах
    evt2 = CheckEvtsForAmplitudeThreshold(vars.thresholdForEEGInVolts)#экземпляр получит значение, только в режиме NeedEpoching
    cndtnlEvtForTargetEpochs = resonance.pipe.filter_event(EvtsMaxAmplitudes, evt2)#оставляем эвенты только тех окон, которые проходят по порогу интервалов
    #берём окна для целевых эпох по отфильтрованным событиям (просмотр максимумов амплитуд (отбрасывание артефактных окон) по всему окну)
    eeg_wndwzd_ = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, cndtnlEvtForTargetEpochs, vars.window_size, vars.window_shift)
    baselinedEpoch_ = resonance.pipe.baseline(eeg_wndwzd_, slice(vars.baseline_begin_offset, vars.baseline_end_offset))

    #поскольку визуализации объекта окон пока нет, то преобразуем к эвенту чтобы посмотреть выхлоп во вьювере
    exhaustEvt = resonance.pipe.transform_to_event(baselinedEpoch_, makeEvtAndAddTargetEpochToList)
    resonance.createOutput(exhaustEvt, 'TargetEpochEvt')