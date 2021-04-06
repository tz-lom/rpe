import resonance
import resonance.pipe
import resonance.cross
import scipy.signal as sp_sig
import numpy as np
from pathlib import Path
import os

Cntr = 0
#PATH = Path('logs/')
#if not os.path.exists(PATH): os.makedirs(PATH)

def online_processing_0():
    import resonance
    import resonance.pipe
    import scipy.signal as sp_sig
    eeg = resonance.input(0)
    cut_off_frequency = 35  # change this to 3 to suppress signal
    #low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')
    '''
    bandstop_frequency = 50
    w0 = [(bandstop_frequency - 0.5) / (eeg.SI.samplingRate / 2), (bandstop_frequency + 0.5) / (eeg.SI.samplingRate / 2)]
    bandstop_filter = sp_sig.butter(4, w0, btype='bandstop')
    eeg_filtered = resonance.pipe.filter(eeg, bandstop_filter)
    resonance.createOutput(eeg_filtered, 'out')
    '''
    highpass_frequency = 0.1
    highpass_filter = sp_sig.butter(4, highpass_frequency, 'hp', fs=eeg.SI.samplingRate)
    eeg_filtered = resonance.pipe.filter(eeg, highpass_filter)
    resonance.createOutput(eeg_filtered, 'out')


def makeEvent(block):
    return np.max(block)


def online_processing_1():
    eeg = resonance.input(0)

    cut_off_frequency = 30
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')

    eeg_filtered = resonance.pipe.filter(eeg, low_pass_filter)
    events = resonance.input(1)

    window_size = 500
    eeg_windowized = resonance.cross.windowize_by_events(eeg_filtered, events, window_size)

    baselined = resonance.pipe.baseline(eeg_windowized)

    result = resonance.pipe.transform_to_event(baselined, makeEvent)

    resonance.createOutput(result, 'out')


def online_processing_2():
    eeg = resonance.input(0)

    cut_off_frequency = 30
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')

    eeg_filtered = resonance.pipe.filter(eeg, low_pass_filter)
    events = resonance.input(1)

    window_size = 500
    window_shift = -window_size
    eeg_windowized = resonance.cross.windowize_by_events(eeg_filtered, events, window_size, window_shift)
    baselined = resonance.pipe.baseline(eeg_windowized)
    result = resonance.pipe.transform_to_event(baselined, makeEvent)
    resonance.createOutput(result, 'out')


def online_processing_3():
    import resonance.pipe
    import scipy.signal as sp_sig
    eeg = resonance.input(0)

    cut_off_frequency = 30
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')

    # eeg_data = np.array(eeg)
    '''
    eeg_data = eeg[:, 0]
    eeg_data = np.reshape(eeg_data, (eeg.shape[0], 1))
    '''
    eeg_filtered = resonance.pipe.filter(eeg, low_pass_filter)

    # так можно сделать расчёт референта, например

    spatial_filter = np.eye(eeg_filtered.SI.channels)  # единичная матрица

    # [ 1  0  0 ]
    # [ 0  1  0 ]
    # [ 0  0  1 ]
    spatial_filter[0, ...] = -1  # первый канал вычитаем из остальных
    # пример, если каналы - строки
    # [-1  0  0 ]
    # [-1  1  0 ]
    # [-1  0  1 ]
    #если каналы - столбцы, как у нас, то
    # [-1  -1  -1 ]
    # [ 0   1   0 ]
    # [ 0   0   1 ]
    spatial_filter = spatial_filter[..., 1:]  # первый канал это референт, убираем его из списка каналов
    # пример, если каналы - строки
    # [-1  1  0 ]
    # [-1  0  1 ]
    # если каналы - столбцы, как у нас, то
    # [-1  -1 ]
    # [ 1   0 ]
    # [ 0   1 ]
    eeg_referenced = resonance.pipe.spatial(eeg_filtered, spatial_filter)

    events = resonance.input(1)

    window_size = 250
    eeg_windowized = resonance.cross.windowize_by_events(eeg_referenced, events, window_size)

    baseline_begin_offset = 0
    baseline_end_offset = 250
    baselined = resonance.pipe.baseline(eeg_windowized, slice(baseline_begin_offset, baseline_end_offset))
    baselined = baselined + 10
    resonance.createOutput(baselined, 'out')

    # как делать мат операции?

    # def your_operation(channels):
    #     return channels + 10
    #
    # result = resonance.pipe.transform_channels(baselined, baselined.SI.channels, your_operation)
    # resonance.createOutput(result, 'after_operation')


def online_processing_4():
    import resonance.pipe
    import scipy.signal as sp_sig

    eeg = resonance.input(0)
    events = resonance.input(1)
    '''
    с фильтрами надо позже разобраться. Постоянку чем резать, пока не понятно
    и нужен ли 50Гц режектор, если полоса ограничена 25Гц, скажем?
    '''
    cut_off_frequency = 25
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')
    eeg_filtered = resonance.pipe.filter(eeg, low_pass_filter)

    resonance.createOutput(eeg_filtered, 'out')

    window_size = 500
    window_shift = -250
    baseline_begin_offset = 0
    baseline_end_offset = 250

    def rule1(evt):
        return evt == "1"

    def rule2(evt):
        return evt == "2"
    #это позволяет на входящий эвент определённого типа реагировать созданием эвентного потока
    cmd1 = resonance.pipe.filter_event(events, rule1)
    resonance.createOutput(cmd1, 'Evtout1')

    # это позволяет на входящий эвент определённого типа реагировать взятием эпохи
    cmd2 = resonance.pipe.filter_event(events, rule2)

    eeg_windowized = resonance.cross.windowize_by_events(eeg_filtered, cmd2, window_size, window_shift)
    baselined = resonance.pipe.baseline(eeg_windowized, slice(baseline_begin_offset, baseline_end_offset))
    result_ = resonance.pipe.transform_to_event(baselined, makeEvent)
    resonance.createOutput(result_, 'Evtout2')

    cmd3 = resonance.pipe.filter_event(events, lambda evt: not (rule1(evt) or rule2(evt)))
    resonance.createOutput(cmd3, 'Evtout3')


def online_processing_5():
    import numpy as np

    eeg = resonance.input(0)

    global Cntr
    if Cntr == 0:
        ts = np.array(eeg.TS)
        np.savetxt('d:/Projects/BCI_EyeLines_Online_2020/rpe/test.txt', ts)
    Cntr = Cntr + 1

    eeg_windows = resonance.pipe.windowizer(eeg, 100, 100)
    #eeg_windows = np.array(eeg_windows)
    as_events = resonance.pipe.transform_to_event(eeg_windows, makeEvent)
    #as_events = np.array(as_events)
    conditional = resonance.pipe.filter_event(as_events, lambda evt: float(evt) > 0)

    resonance.createOutput(conditional, 'out')

#селективная фильтрация каналов + корректировка ЭЭГ на референт
def online_processing_6():
    import resonance.pipe
    import scipy.signal as sp_sig

    allData = resonance.input(0)
    events = resonance.input(1)

    np_eye = np.eye(allData.SI.channels)  # единичная матрица - основа пространственной фильтрации

    sptl_filtr0 = np_eye[..., :-1]  # 2 последний канал - эвентный канал, убираем его из списка каналов (убираем из единичной матрицы крайний столбец)
    woEvtChnlData = resonance.pipe.spatial(allData, sptl_filtr0)

    bandstop_frequency = 50 # применяем режектор 50Гц для всех данных, кроме эвентного канала
    w0 = [(bandstop_frequency - 0.5) / (allData.SI.samplingRate / 2), (bandstop_frequency + 0.5) / (allData.SI.samplingRate / 2)]
    bandstop_filter = sp_sig.butter(4, w0, btype='bandstop')
    woEvtChnldata_Notchfiltered = resonance.pipe.filter(woEvtChnlData, bandstop_filter)
    resonance.createOutput(woEvtChnldata_Notchfiltered, 'AllData_Notch_Filtered')

    np_eye2 = np.eye(allData.SI.channels-1)  # единичная матрица для всех каналов минус эвентный канал

    sptl_filtr = np_eye2[..., :-1]  #оставляем только ЭЭГ, убираем последний канал - это ЭМГ (убираем из единичной матрицы крайний столбец)
    eegData = resonance.pipe.spatial(woEvtChnldata_Notchfiltered, sptl_filtr)
    sptl_filtr2 = np_eye2[..., -1:] #оставляем последний столбец - для отдельной фильтрации ЭМГ канала
    emgData = resonance.pipe.spatial(woEvtChnldata_Notchfiltered, sptl_filtr2)

    cut_off_frequency = 35
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / allData.SI.samplingRate * 2, btype='low')
    eeg_LPfiltered = resonance.pipe.filter(eegData, low_pass_filter)
    resonance.createOutput(eeg_LPfiltered, 'EEG_Notch_LP_Filtered')

    highpass_frequency = 0.1
    highpass_filter = sp_sig.butter(4, highpass_frequency, 'hp', fs=allData.SI.samplingRate)
    eeg_LPHPfiltered = resonance.pipe.filter(eeg_LPfiltered, highpass_filter)
    resonance.createOutput(eeg_LPHPfiltered, 'EEG_Notch_LP_HPFiltered')

    #уже фильтрованную ЭЭГ корректируем на референт, в качестве которого возьмём 1й канал
    np_eye3 = np.eye(allData.SI.channels - 2)  # единичная матрица для всех каналов минус эвентный канал и минус ЭМГ
    np_eye3[0, ...] = -1  # готовим матрицу, чтобы 1й канал вычесть из остальных (нулевая строка = -1)
    sptl_filtr4 = np_eye3[...,1:]  # первый канал - это референт, убираем его из списка каналов (Убираем первый столбец матрицы)
    eeg_Filtered_Referenced = resonance.pipe.spatial(eeg_LPHPfiltered, sptl_filtr4)
    resonance.createOutput(eeg_LPHPfiltered, 'EEG_Filtered_Referenced')

    #для ЭМГ применим только фильтр высоких частот
    highpass_frequency = 5
    highpass_filter = sp_sig.butter(4, highpass_frequency, 'hp', fs=allData.SI.samplingRate)
    emg_HPfiltered = resonance.pipe.filter(emgData, highpass_filter)
    resonance.createOutput(emg_HPfiltered, 'EMG_Notch_HPFiltered')

