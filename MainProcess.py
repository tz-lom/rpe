import resonance
import resonance.pipe
import resonance.cross
import scipy.signal as sp_sig
import numpy as np
import datetime
import time
import threading
from enum import Enum
from enum import IntEnum
from mne.filter import resample
from sklearn.model_selection import train_test_split
from train_clf import train_model
import os
from utils import read_params, write_params
import glob
from keras.models import load_model
import vars
import tkinter
import sys
import pickle

PeriodicProc = None
ValueForSendEvt = 0.0
NeedSendEvt = False

class WS(IntEnum):
    None_ = 0
    NeedEpochingForTraining = 1
    NeedPrepareEpochs = 2
    NeedTraining = 3
    PrepareForClassifierApply = 4
    TooFewEpochsForTraining = 5
    ClassifierApply = 6
    SettingsChanged = 7
    Reserved = 8
    Handshake = 9 #запрос от стимпроги, на который предполагается дать ответ отюда туда для подтверждения что связь ок
    ModelTrainedOK = 10
    ModelPrepared = 11
    ErrorModelNotPrepared = 13

Trained_model = None
WorkState = WS.None_
TSFirstDataValue = 0
predictions = [0]

X_Epochs = []
Y_Epochs = []

X_Tr = []
Y_Tr = []

X_Tst = []
Y_Tst = []

ClassifierPrepared = False

logfileTargetEpochs = os.path.dirname(__file__) + "/logs/targetepochs.txt"
logfileUnTargetEpochs = os.path.dirname(__file__) + "/logs/untargetepochs.txt"
logFileNamePreparedEpochs = os.path.dirname(__file__) + "/logs/AllEpochsTS.txt"
logFileNamePreparedEpochs2 = os.path.dirname(__file__) + "/logs/PrepareEpochs.txt"
logFileNamePreparedEpochs3 = os.path.dirname(__file__) + "/logs/PreparedEpochs.txt"
logfileModelParams = os.path.dirname(__file__) + "/logs/ModelParams.txt"
logfileModelTrainedRes = os.path.dirname(__file__) + "/logs/ModelTrainedRes.txt"
logFileModelPredictions = os.path.dirname(__file__) + "/logs/ModelPredictions.txt"
dataFileXTrEpochs = os.path.dirname(__file__) + "/logs/dataFileXTrEpochs.pkl"
dataFileYTrEpochs = os.path.dirname(__file__) + "/logs/dataFileYTrEpochs.pkl"
dataFileXTstEpochs = os.path.dirname(__file__) + "/logs/dataFileXTstEpochs.pkl"
dataFileYTstEpochs = os.path.dirname(__file__) + "/logs/dataFileYTstEpochs.pkl"

# селективная фильтрация каналов + корректировка ЭЭГ на референт
def FiltrationAndRefCorrection(allData):
    np_eye = np.eye(allData.SI.channels)  # единичная матрица - основа пространственной фильтрации

    sptl_filtr0 = np_eye[...,:-1]  # 2 последний канал - эвентный канал, убираем его из списка каналов (убираем из единичной матрицы крайний столбец)
    woEvtChnlData = resonance.pipe.spatial(allData, sptl_filtr0)

    bandstop_frequency = vars.alldata_bandstop_frequency  # применяем режектор 50Гц для всех данных (эвентный канал выкинули)
    w0 = [(bandstop_frequency - 0.5) / (allData.SI.samplingRate / 2),(bandstop_frequency + 0.5) / (allData.SI.samplingRate / 2)]#50+-0.5Гц
    bandstop_filter = sp_sig.butter(4, w0, btype='bandstop')
    woEvtChnldata_Notchfiltered = resonance.pipe.filter(woEvtChnlData, bandstop_filter)
    #resonance.createOutput(woEvtChnldata_Notchfiltered, 'AllData_Notch_Filtered')

    np_eye2 = np.eye(allData.SI.channels - 1)  # единичная матрица для всех каналов минус эвентный канал

    sptl_filtr = np_eye2[...,:-1]  # оставляем только ЭЭГ, убираем последний канал - это ЭМГ (убираем из единичной матрицы крайний столбец)
    eegData_Notchfiltered = resonance.pipe.spatial(woEvtChnldata_Notchfiltered, sptl_filtr)
    sptl_filtr2 = np_eye2[..., -1:]  # оставляем последний столбец - для отдельной фильтрации ЭМГ канала
    emgData_Notchfiltered = resonance.pipe.spatial(woEvtChnldata_Notchfiltered, sptl_filtr2)

    cut_off_frequency = vars.eeg_cut_off_frequency
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / allData.SI.samplingRate * 2, btype='low')
    eeg_NotchLPfiltered = resonance.pipe.filter(eegData_Notchfiltered, low_pass_filter)
    #resonance.createOutput(eeg_NotchLPfiltered, 'EEG_Notch_LP_Filtered')

    highpass_frequency = vars.eeg_highpass_frequency
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
    highpass_frequency = vars.emg_highpass_frequency
    highpass_filter = sp_sig.butter(4, highpass_frequency, 'hp', fs=allData.SI.samplingRate)
    emg_HPfiltered = resonance.pipe.filter(emgData_Notchfiltered, highpass_filter)
    #resonance.createOutput(emg_HPfiltered, 'EMG_Notch_HPFiltered')
    return eeg_Filtered_Referenced, emg_HPfiltered

def workerModelTraining():
    """thread worker function"""
    global X_Tr, Y_Tr
    global Trained_model
    Trained_model, val_auc = train_model(X_Tr, Y_Tr)
    val_auc = np.array(val_auc)

    cur_datetime = datetime.datetime.now()
    cur_datetime_string = cur_datetime.strftime('%d-%m-%y_%H-%M-%S')
    Trained_model.save(os.path.join(vars.path_to_current_model, 'nn_%s.hdf5' % cur_datetime_string))

    #logFileName = os.path.join(os.path.dirname(__file__), "train_model.txt")
    global WorkState
    if Trained_model is not None:
        with open(logfileModelTrainedRes, "w+") as f:
            # f.write("val_auc:\n")
            np.savetxt(f, [val_auc], fmt='%.18g')
        WorkState = WS.ModelTrainedOK

        global ValueForSendEvt
        global NeedSendEvt
        ValueForSendEvt = float(WorkState)
        NeedSendEvt = True
    return

def loadModelThread():
    global Trained_model
    global ClassifierPrepared
    global WorkState
    global ValueForSendEvt
    global NeedSendEvt
    params = read_params('config.json')
    path_to_models_dir = params['path_to_models_dir']
    path_to_model = os.path.dirname(__file__) + path_to_models_dir + '/**/*.hdf5'
    filename_list = glob.glob(path_to_model)
    if len(filename_list) > 0:
        filename_ = filename_list[-1]
        Trained_model = load_model(filename_)

    if Trained_model is None:
        ClassifierPrepared = False
    else:
        ClassifierPrepared = True
        WorkState = WS.ModelPrepared #
        ValueForSendEvt = float(WorkState)
        NeedSendEvt = True

    return

def Init():
    global logfileUnTargetEpochs, logfileTargetEpochs
    #пусть по умолчанию будут созданы файлы с нулями
    with open(logfileUnTargetEpochs, "w+") as f:
        np.savetxt(f, [0], fmt='%-10d', delimiter=',', newline='')#-10d - выравнивание по левому краю, 10 полей. На случай если надо затереть большое число
    with open(logfileTargetEpochs, "w+") as f:
        np.savetxt(f, [0], fmt='%-10d', delimiter=',', newline='')

    with open(logFileNamePreparedEpochs3, "w+") as f:
        np.savetxt(f, [0], fmt='%-10d', delimiter=',', newline='')
        np.savetxt(f, [0], fmt='%-10d', delimiter=',', newline='')

    with open(logfileModelParams, "w+") as f:
        np.savetxt(f, [0], fmt='%-10d', delimiter=',', newline='')

    with open(logfileModelTrainedRes, "w+") as f:
        np.savetxt(f, [0], fmt='%-10d', delimiter=',', newline='')

    if os.path.isfile(logFileModelPredictions):
        os.remove(logFileModelPredictions)

    # params = read_params('bin\SStim.json') #Проще править vars.py перед запуском
    #
    # vars.window_size = params['window_size']
    # vars.window_shift = params['window_shift']
    # vars.baseline_begin_offset = params['baseline_begin_offset']
    # vars.baseline_end_offset = params['baseline_end_offset']
    # vars.thresholdForEEGInVolts = params['thresholdForEEGInVolts']  # пороговая амплитуда помехи для оценки кандидата на нецелевую эпоху 500мкВ
    # vars.thresholdForEMGInVolts = params['thresholdForEMGInVolts']  # пороговая амплитуда ЭМГ для оценки кандидата на целевую эпоху
    # vars.intervalEvtToEvtInSec = params['intervalEvtToEvtInSec']  # интервал между кандидатами на целевую/нецелевую эпоху, сек


def makeOutEvent(block):
    global NeedSendEvt
    if NeedSendEvt:
        return ValueForSendEvt
    else:
        return 0
#черезжопный способ генерации события в стимпрогу через флаг, который взводится там, где надо отправить значение в стимпрогу через переменную
def UpdateEvtState(evt):
    global NeedSendEvt
    if NeedSendEvt:
        NeedSendEvt = False #иначе будет отправляться Адноитоже до усрачки
        return True
    else:
        return False

#меняем рабочий статус системы согласно входящим командам от стимпроги
def UpdateWorkState(evt):
    global WorkState
    global NeedSendEvt
    global ValueForSendEvt
    if (evt == "0"):
        WorkState = WS.None_
    elif (evt == "1"):
        WorkState = WS.NeedEpochingForTraining
        ValueForSendEvt = float(WorkState)
        NeedSendEvt = True
    elif (evt == "2"):
        WorkState = WS.NeedPrepareEpochs
        global X_Tr, Y_Tr, X_Tst, Y_Tst
        #почему нет объявления global X_Epochs global Y_Epochs ? Нихрена не понятно !!!!!!!!!!!!!!!!!!!!!!!!!!!!! Уточнить 23.06.2023
        global X_Epochs, Y_Epochs #объявил. Надо смотреть как будет 23.06.2023
        X_Tr, Y_Tr, X_Tst, Y_Tst = EpochsPrepare(X_Epochs, Y_Epochs)
        if os.path.isfile(logFileNamePreparedEpochs):  # удаляем логи со списком подготовленных эпох
            os.remove(logFileNamePreparedEpochs)
        #для контроля эпох в матлабе: открываем там ЭЭГ и текстовый файл с метками времен
        with open(logFileNamePreparedEpochs, "wt") as f:
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
        #logFileName = os.path.join(os.path.dirname(__file__), "PrepareEpochs.txt")

        with open(logFileNamePreparedEpochs3, "wt") as f:
            np.savetxt(f, [len(X_Tr)], fmt='%-10d', delimiter=',', newline='')
            f.write("\n")
            np.savetxt(f, [len(X_Tst)], fmt='%-10d', delimiter=',', newline='')
        if os.path.isfile(logFileNamePreparedEpochs2):
            os.remove(logFileNamePreparedEpochs2)
        with open(logFileNamePreparedEpochs2, "wt") as f:
            f.write("PrepareEpochs\n")
            f.write("TrainingEpochsCnt: ")
            np.savetxt(f, [len(X_Tr)], fmt='%.1d')
            f.write("Y_TrainingEpochs: \n")
            np.savetxt(f, Y_Tr, fmt='%.1d', delimiter=',', newline='')

            f.write("\nTstEpochsCnt: ")
            np.savetxt(f, [len(X_Tst)], fmt='%.1d')
            f.write("Y_TstEpochs: \n")
            np.savetxt(f, Y_Tst, fmt='%.1d', delimiter=',', newline='')

        with open(logfileModelParams, "wt") as f:
            np.savetxt(f, [vars.proportion_of_the_TestDataset], fmt='%-18g', delimiter=',', newline='')
            f.write("\n")
        #сохраним подготовленные эпохи чтобы можно было прочитать из файлов, если что-то пойдёт не так
        with open(dataFileXTrEpochs, 'wb') as output:
            pickle.dump(X_Tr, output, pickle.HIGHEST_PROTOCOL)
        with open(dataFileYTrEpochs, 'wb') as output2:
            pickle.dump(Y_Tr, output2, pickle.HIGHEST_PROTOCOL)

        with open(dataFileXTstEpochs, 'wb') as output3:
            pickle.dump(X_Tst, output3, pickle.HIGHEST_PROTOCOL)
        with open(dataFileYTstEpochs, 'wb') as output4:
            pickle.dump(Y_Tst, output4, pickle.HIGHEST_PROTOCOL)

        #лучше потом в управляющую прогу заслать значения Как?????!!!!! 23.06.2023
        #пока никак. Читаю из логов, созданных тут. Тупо, но пока надо хоть как-то заставить это работать. 09.07.2023
        ValueForSendEvt = float(WorkState)#при реальной работе с ээг и завершения комплектации эпох, падает стимпрога, что сильно расстраивает. Я не знаю почему такое происходит
        NeedSendEvt = True #пока придётся просто выключить отправку значения в стимпрогу 20.07.2023
    elif (evt == "3"):
        if len(Y_Tr) == 0:
            # читаем эпохи из файлов
            with open(dataFileXTrEpochs, 'rb') as pfile1:
                X_Tr = pickle.load(pfile1)
            with open(dataFileYTrEpochs, 'rb') as pfile2:
                Y_Tr = pickle.load(pfile2)
            with open(dataFileXTstEpochs, 'rb') as pfile3:
                X_Tst = pickle.load(pfile3)
            with open(dataFileYTstEpochs, 'rb') as pfile4:
                Y_Tst = pickle.load(pfile4)

        #Надо проверять, если ли возможность обучать классификатор
        if len(Y_Tr) > 10: #!! вынести в настройки - минимальное кол-во обучающих эпох
            WorkState = WS.NeedTraining
            ValueForSendEvt = float(WorkState)
            NeedSendEvt = True
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
            task = threading.Thread(target=workerModelTraining)
            task.start()
        else:
            WorkState = WS.TooFewEpochsForTraining
            ValueForSendEvt = float(WorkState)
            NeedSendEvt = True
            return False
    elif (evt == "4"):
        WorkState = WS.PrepareForClassifierApply
        task = threading.Thread(target = loadModelThread)#загрузка обученной модели из файла
        task.start()

    elif (evt == "6"):
        WorkState = WS.ClassifierApply
        ValueForSendEvt = float(WorkState)
        NeedSendEvt = True

   # elif (evt == "7"):
   #     '''считывание настроек из файла, в который они сохраняются из полей контролов стим проги
   #     Это не работает!!! Видимо onlineprocessing вызывается один раз раньше и всё на этом даже если команда на чтение приходит, то пофиг
   #     надо делать init, пихать в onlineprocessing и смотреть что будет'''
   #     tmpVar = WorkState
   #     #WorkState = WS.SettingsChanged
   #     params = read_params('bin\SStim.json')
   #     vars.window_size = params['window_size']
   #     vars.window_shift = params['window_shift']
   #     vars.baseline_begin_offset = params['baseline_begin_offset']
   #     vars.baseline_end_offset = params['baseline_end_offset']
   #     vars.thresholdForEEGInVolts = params['thresholdForEEGInVolts']  # пороговая амплитуда помехи для оценки кандидата на нецелевую эпоху 500мкВ
   #     vars.thresholdForEMGInVolts = params['thresholdForEMGInVolts']  # пороговая амплитуда ЭМГ для оценки кандидата на целевую эпоху
   #     vars.intervalEvtToEvtInSec = params['intervalEvtToEvtInSec']  # интервал между кандидатами на целевую/нецелевую эпоху, сек
        #WorkState = tmpVar

    elif (evt == "8"):
        params = read_params('bin\SStim.json')
        param1 = params['intervalEvtToEvtInSec']
        WorkState = param1

    elif (evt == "9"):#для верификации линка между резонансом и стимпрогой. Если ок, то там загорается зелёная лампочка
        WorkState = WS.Handshake
        ValueForSendEvt = float(WorkState)
        NeedSendEvt = True


    return True

#просмотр максимумов шинкованных микроокон и выбор кандидатов на взятие окна для нецелевого класса
class GetEventsForUntargetedEpochs:
    def __init__(self, threshold, delay):
        self._threshold = threshold
        self._delay = delay
        self._last_event = -np.Infinity

    def __call__(self, evt):
        global ValueForSendEvt
        global NeedSendEvt
        if WorkState == WS.NeedEpochingForTraining:
            #смотрим амплитуды по порогу
            if float(abs(evt) < self._threshold):
                current_ts = evt.timestamps[-1]
                if self._last_event < current_ts - self._delay * 1e9:  # minimal delay between events = T seconds
                    self._last_event = current_ts #берём подходящий по амплитуде эвент, только если он по времени больше заданного интервала
                    return True
        elif WorkState == WS.ClassifierApply:
            if float(abs(evt) < self._threshold):
                # ValueForSendEvt = float(15)#Если кандидатная эпоха ok
                # NeedSendEvt = True
                return True
            else:
                ValueForSendEvt = float(14)#Если кандидатная эпоха при применении классификатора подверглась режекции по амплитуде ЭЭГ
                NeedSendEvt = True
                return False
        return False

class GetFirstTSData:
    def __init__(self, needTS):
        self._needTS = needTS
    def __call__(self, evt):
        if self._needTS == True:
            global TSFirstDataValue
            TSFirstDataValue = evt.timestamps[0]
            #self._needTS = False
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
    resample_to = vars.resample_data_to
    sourceSR = 500  # переделать потом! сделать класс и передавать SR в него в init
    resampledEpochs = []
    for item in Xepochs:
        dataResampled = _resample(np.array(item), sourceSR, resample_to)
        resampledEpochs.append(dataResampled)
    X = np.array(resampledEpochs)
    X = X.transpose([0, 2, 1])
    x_tr_val_ind, x_tst_ind, y_tr_val, y_tst = train_test_split(range(X.shape[0]), Yepochs, test_size=vars.proportion_of_the_TestDataset, stratify=Yepochs)
    x_tr_val = X[x_tr_val_ind, ...]
    x_tst = X[x_tst_ind, ...]
    return x_tr_val, y_tr_val, x_tst, y_tst

def workerModelApplying(block):

    global Trained_model
    global predictions

    global ValueForSendEvt
    global NeedSendEvt
    resample_to = vars.resample_data_to
    sourceSR = 500  # переделать потом! сделать класс и передавать SR в него в init
    dataResampled = _resample(np.array(block), sourceSR, resample_to)
    dataResampled = dataResampled.transpose([1, 0])  # исходно строки - отсчёта, столбцы - каналы. Надо наоборот
    dataResampled = dataResampled[np.newaxis, np.newaxis, :, :]

    predictions = Trained_model.predict(dataResampled)[:, 1]

    if predictions[0] > vars.model_predictions_threshold:
        ValueForSendEvt = float(12)  # кодовое значение из Резонанса, когда классификатор сработал/в стимпроге надо что-то сделать на это событие
        NeedSendEvt = True

    # все значения предсказания модели сохраняем в лог вместе с временем блока
    with open(logFileModelPredictions, "a+") as f:
        np.savetxt(f, np.column_stack([block.timestamps[0], predictions[0]]), fmt='%-18f', delimiter=' ',
                   newline='\n')

    return

def makeEvtAndAddUntargetEpochToList(block):
    if WorkState == WS.NeedEpochingForTraining:
        global X_Epochs
        global Y_Epochs
        global ValueForSendEvt
        global NeedSendEvt

        _delay = 2  # чтобы не было коллизий с крайней эпохой; выдерживается интервал между эпохами (нецелевыми/целевыми)!!! вынести в глобальную переменную, что ли?

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
            X_Epochs.append(block)#тут добавляется самый первый блок Следующий будет сравниваться с ним по времени
            Y_Epochs.append(0)

        with open(logfileUnTargetEpochs, "w+") as f:
            np.savetxt(f, [len(np.where(np.array(Y_Epochs) == 0)[0])], fmt='%-10d', delimiter=',', newline='')

        return len(np.where(np.array(Y_Epochs) == 0)[0])#если прошёл порог амплитуд, но не прошёл порог по интервалу, то выводится текущий счётчик нецелевых эпох
        #значение счётчика увеличивается только если эпоха проходит порог амплитуд и интервалов
    elif WorkState == WS.NeedPrepareEpochs:
        return 123

    elif WorkState == WS.ClassifierApply:
        global ClassifierPrepared
        if ClassifierPrepared:

            # task = threading.Thread(target=workerModelApplying(block))
            # task.start()

            global Trained_model
            global predictions

            global ValueForSendEvt
            global NeedSendEvt
            resample_to = vars.resample_data_to
            sourceSR = 500  # переделать потом! сделать класс и передавать SR в него в init
            dataResampled = _resample(np.array(block), sourceSR, resample_to)
            dataResampled = dataResampled.transpose([1, 0])  # исходно строки - отсчёта, столбцы - каналы. Надо наоборот
            dataResampled = dataResampled[np.newaxis, np.newaxis, :, :]

            predictions = Trained_model.predict(dataResampled)[:, 1]
            #predictions = Trained_model(dataResampled)[:, 1]

            if predictions[0] > vars.model_predictions_threshold:
                ValueForSendEvt = float(12)  # кодовое значение из Резонанса, когда классификатор сработал/в стимпроге надо что-то сделать на это событие
                NeedSendEvt = True

            # ValueForSendEvt = float(predictions[0] * 100)  # кодовое значение из Резонанса, когда классификатор сработал/в стимпроге надо что-то сделать на это событие
            # NeedSendEvt = True

            # все значения предсказания модели сохраняем в лог вместе с временем блока
            # with open(logFileModelPredictions, "a+") as f:
            #     np.savetxt(f, np.column_stack([block.timestamps[0], predictions[0]]), fmt='%-18f', delimiter=' ',
            #                newline='\n')


            # resample_to = vars.resample_data_to
            # sourceSR = 500  # переделать потом! сделать класс и передавать SR в него в init
            # dataResampled = _resample(np.array(block), sourceSR, resample_to)
            # dataResampled = dataResampled.transpose([1, 0])#исходно строки - отсчёта, столбцы - каналы. Надо наоборот
            # dataResampled = dataResampled[np.newaxis, np.newaxis, :, :]
            # global Trained_model
            # predictions = Trained_model.predict(dataResampled)[:, 1]
            #
            # if predictions[0] > vars.model_predictions_threshold:
            #     ValueForSendEvt = float(12)#кодовое значение из Резонанса, когда классификатор сработал/в стимпроге надо что-то сделать на это событие
            #     NeedSendEvt = True
            #
            # #все значения предсказания модели сохраняем в лог вместе с временем блока
            # with open(logFileModelPredictions, "a+") as f:
            #     np.savetxt(f, np.column_stack([block.timestamps[0], predictions[0]]), fmt='%-18f', delimiter=' ', newline='\n')

            return predictions[0]

        else:
            WorkState == WS.ErrorModelNotPrepared
            ValueForSendEvt = float(WorkState)
            NeedSendEvt = True
            return 345
    return abs(np.max(block))

def makeEvtAndAddTargetEpochToList(block):
    if WorkState == WS.NeedEpochingForTraining:
        global X_Epochs
        global Y_Epochs

        if (len(X_Epochs) > 0) and (len(Y_Epochs) > 0):
            predEpoch = X_Epochs[-1]
            _delay = 2 #!!! вынести в глобальную переменную, что ли? !!!!! Какая-то хрень. Надо разобраться!!! 23.06.2023
            # если время начала новой эпохи минус время конца эпохи в списке меньше заданного интервала, удаляем данную эпоху из списка
            endDTNewEpoch = block.timestamps[-1]#время конца новой эпохи
            endDTOldEpoch = predEpoch.timestamps[-1]#время конца крайней в списке эпохи
            if endDTNewEpoch - endDTOldEpoch < _delay * 1e9:
                X_Epochs.pop(-1)
                Y_Epochs.pop(-1)

        X_Epochs.append(block)
        Y_Epochs.append(1)

        with open(logfileTargetEpochs, "w+") as f:
            np.savetxt(f, [len(np.where(np.array(Y_Epochs) == 1)[0])], fmt='%-10d', delimiter=',', newline='')

    return len(np.where(np.array(Y_Epochs) == 1)[0])

def online_processing():
    Init() #
    alldata = resonance.input(0)
    eeg_Filtered_Referenced, emg_Notch_HPFiltered = FiltrationAndRefCorrection(alldata)# селективная фильтрация каналов + корректировка ЭЭГ на референт
    resonance.createOutput(eeg_Filtered_Referenced, 'EEG_Filtered_Referenced')
    resonance.createOutput(emg_Notch_HPFiltered, 'EMG_Notch_HPFiltered')

    #просматриваем эвенты и меняем рабочий статус системы согласно входящим командам
    events = resonance.input(1)
    cmd_input = resonance.pipe.filter_event(events, UpdateWorkState)
    #resonance.createOutput(cmd_input, 'EvtWorkState')#входящие команды эхом отправляем в эвентный поток для контроля

    #Формируем эпохи для нецелевого класса
    #просмотр шинкованной ЭЭГ на предмет кандидатов на формирование нецелевых эпох
    #для 32 каналов ЭЭГ + 1 ЭМГ применение классификатора не тянет с меньшими окнами, чем 30,30
    #с 5ю каналами ЭЭГ 10,10 не тянет почти также. 20, 20 уже почти норм.
    eeg_windows = resonance.pipe.windowizer(eeg_Filtered_Referenced, 25, 25)#шинкуем ЭЭГ на мелкие окна 50мс, если 25 отсчётов при Fs=500Гц
    eeg_as_events = resonance.pipe.transform_to_event(eeg_windows, makeEvent)

    #эта хрень будет апдейтить ситуацию, полузуясь событиями прихода блоков как тиками таймера и генерить на выход в стимпрогу значение, если надо
    evtOut = resonance.pipe.transform_to_event(eeg_windows, makeOutEvent)#если флаг взведён, выставим значение, которое будем отправлять
    cmd_output = resonance.pipe.filter_event(evtOut, UpdateEvtState)#если флаг взведём генерим событие и сбрасываем флаг
    resonance.createOutput(cmd_output, 'EvtOut1')
    # evt11 = GetFirstTSData(True)
    # eeg_as_events_ = resonance.pipe.filter_event(eeg_as_events, evt11)#второй аргумент-окно, которое выдаётся из недр резонанса, если предыдущ. ф-ция сработала (выдала true)
    # resonance.createOutput(eeg_as_events_,'EvtOut2')

    eeg_windowized = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, eeg_as_events, vars.window_size, vars.window_shift)
    EvtsMaxAmplitudes = resonance.pipe.transform_to_event(eeg_windowized, makeEvent)  # считаем максимумы амплитуд во взятых окнах
    #экземпляр получит значение в режиме NeedEpoching с оценкой порогов ампл и интервала. В режиме ClassifierNeedApplay смотрим только порог ампл и экземпляр получает значение
    evt = GetEventsForUntargetedEpochs(vars.thresholdForEEGInVolts, vars.intrvlEvtToEvtInSecForUntargetedEpochs)#
    cndtnlEvtForUntargetEpochs = resonance.pipe.filter_event(EvtsMaxAmplitudes, evt)#оставляем эвенты только тех окон, которые проходят по порогу амплитуд и/или интервалов (оценка ранее)
    #resonance.createOutput(cndtnlEvtForUntargetEpochs, 'EvtCndtnlForUntargetEpochs')
    #берём окна для нецелевых эпох по отфильтрованным событиям (просмотр максимумов амплитуд (отбрасывание артефактных окон) по всему окну, интервалов между окнами)
    eeg_windowized_ = resonance.cross.windowize_by_events(eeg_Filtered_Referenced, cndtnlEvtForUntargetEpochs, vars.window_size, vars.window_shift)
    baselinedEpoch = resonance.pipe.baseline(eeg_windowized_, slice(vars.baseline_begin_offset, vars.baseline_end_offset))
    #поскольку визуализации объекта окон пока нет, то преобразуем к эвенту чтобы посмотреть выхлоп во вьювере
    exhaustEvt = resonance.pipe.transform_to_event(baselinedEpoch, makeEvtAndAddUntargetEpochToList)
    resonance.createOutput(exhaustEvt, 'UntargetEpochEvt')
    #та же история взятия эпох для подачи обученному классификатору, только тогда intervalEvtToEvtInSec=0

    # Формируем целевые эпохи на основе порога амплитуды ЭМГ, интервалов между эвентами и безартефактности взятого окна ЭЭГ
    emg_windows = resonance.pipe.windowizer(emg_Notch_HPFiltered, 25, 25)#шинкуем ЭМГ на мелкие окна
    emg_as_events = resonance.pipe.transform_to_event(emg_windows, makeEvent)
    #intervalEvtToEvtInSec = 2.8
    evt1 = GetEventsForTargetedEpochs(vars.thresholdForEMGInVolts, vars.intrvlEvtToEvtInSecForTargetedEpochs)#экземпляр получит значение, только в режиме NeedEpoching
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


#эта хрень нужна для реализации оффлайна, т.к. я не вкурил как делать оффлайн через Резонанс. Там для этого нужны инструменты. Например, открытие записанного файла с ЭЭГ
#с метками в виде эмуляции потока как от актичампа. Тогда можно было бы работать. А щас приходится выкручиваться. 11.07.2023
#Для работы в оффлайне запускается этот модуль в PyCharm, который подхватывает main. Тут создаётся окно с основной функциональностью и интерфейсом пользователя:
#создание рабочего цикла конечного автомата и ф-ций, которые запускаются также как при работе в окне quickStimulus
#Для работы в оффлайне вместо quickStimulus запускам этот модуль, нажимаем старт и далее работаем без стимпроги: выбираем файлы данных, обучаем, верифицируем, сохраняем модель.
# Всё должно работать также как в онлайне, только без quickStimulus и без стимпроги, т.к. связи между MainProcess и стимпрогой без интерфейса Резонанса не будет
# Можно запускать вхолостую генератор сигналов с установкой всех событий. Ради самих событий и работать через стимпрогу, но это изврат
#Ещё обнаружил, Что Резонанс тормозится, если сделать отдельный поток и там каждый 10мс выдавать в консоль что-то. Жопа.
#Поток для генерации событий и движухи по конечному автомату оставлю только для оффлайна. В онлайне буду пользоваться событиями прихода очередного блока при шинковании данных с устройства
class tmrProc():
    def __init__(self):
        self._cur_time = None
        self.periodTm1 = 0.01 #период вызова нужной ф-ции в отдельном потоке в сек
        self.running = True;

    def evtUpdate(self, params = 0):
        #sys.stdout.write('({}) foo\n'.format(datetime.datetime.now()))
        a=0

    def __call__(self):
        self._cur_time = time.perf_counter()
        while self.running:
            if time.perf_counter() - self._cur_time > self.periodTm1:
                self.evtUpdate()
                self._cur_time = time.perf_counter()

def main():
    print('Hello!')
    def wnd_close():
        global running
        if tkinter.messagebox.askokcancel("Quit", "Do you want to quit?"):
            global periodicProc
            if PeriodicProc is not None:
                PeriodicProc.running = False  # turn off while loop
            wnd.destroy()
            print("Window closed")

    def btnStart_click():
        lbl.configure(text="Starting...")
        global PeriodicProc
        if PeriodicProc is None:
            PeriodicProc = tmrProc()
            # periodicProc.periodTm1 = 2
            tmr = threading.Thread(target=PeriodicProc)
            # tmr.daemon = True
            tmr.start()

    def btnStop_click():
        global periodicProc
        if PeriodicProc is not None:
            PeriodicProc.running = False  # turn off while loop

    wnd = tkinter.Tk()
    wnd.protocol("WM_DELETE_WINDOW", wnd_close)
    wnd.title("Welcome to PythonRu application")
    window_height = 600
    window_width = 800
    screen_width = wnd.winfo_screenwidth()
    screen_height = wnd.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (window_width / 2))
    y_cordinate = int((screen_height / 2) - (window_height / 2))
    wnd.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))

    lbl = tkinter.Label(wnd, text="Stim Prog", font=("Arial Bold", 12))
    lbl.grid(column=0, row=0)
    btnStart = tkinter.Button(wnd, text="Start", command=btnStart_click)
    btnStart.grid(column=1, row=0)
    btnStop = tkinter.Button(wnd, text="Stop", command=btnStop_click)
    btnStop.grid(column=2, row=0)

    wnd.mainloop()

if __name__ == '__main__':
    main()