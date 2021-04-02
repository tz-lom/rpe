import resonance
import resonance.pipe
import resonance.cross
import scipy.signal as sp_sig
import numpy as np

def online_processing_example2():
    import resonance
    import resonance.pipe
    import scipy.signal as sp_sig
    eeg = resonance.input(0) 
    cut_off_frequency = 30  # change this to 3 to suppress signal
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')
    eeg_filtered = resonance.pipe.filter(eeg, low_pass_filter)
    resonance.createOutput(eeg_filtered, 'out')

def makeEvent(block):
    return np.max(block)

def online_processing_old():
    eeg = resonance.input(0)

    cut_off_frequency = 30
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')

    eeg_filtered = resonance.pipe.filter(eeg, low_pass_filter)
    events = resonance.input(1)

    window_size = 500
    eeg_windowized = resonance.cross.windowize_by_events(eeg_filtered, events, window_size)

    baseline_begin_offset = 0
    baseline_end_offset = 500
    baselined = resonance.pipe.baseline(eeg_windowized, baseline_begin_offset, baseline_end_offset)

    result = resonance.pipe.transform_to_event(baselined, makeEvent)

    resonance.createOutput(result, 'out')
    
    
def online_processing_v3():
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


def online_processing3():
    import resonance.pipe
    import scipy.signal as sp_sig
    eeg = resonance.input(0)

    cut_off_frequency = 30
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')

   # eeg_data = np.array(eeg)
    eeg_data = eeg[:, 0]
    eeg_data = np.reshape(eeg_data, (eeg.shape[0], 1))
    eeg_filtered = resonance.pipe.filter(eeg_data, low_pass_filter)


    # так можно сделать рассчёт земли например

    spatial_filter = np.eye(eeg_filtered.SI.channels)  # диагональная матрица
    # [ 1  0  0 ]
    # [ 0  1  0 ]
    # [ 0  0  1 ]
    spatial_filter[0, ...] = -1  # первый канал вычитаем из остальных
    # [-1  0  0 ]
    # [-1  1  0 ]
    # [-1  0  1 ]
    spatial_filter = spatial_filter[..., 1:]  # первый канал это земля, убираем его из списка каналов
    # [-1  1  0 ]
    # [-1  0  1 ]

    eeg_referenced = resonance.pipe.spatial(eeg_filtered, spatial_filter)

    events = resonance.input(1)

    window_size = 250
    eeg_windowized = resonance.cross.windowize_by_events(eeg_referenced, events, window_size)

    baseline_begin_offset = 0
    baseline_end_offset = 250
    baselined = resonance.pipe.baseline(eeg_windowized, slice(baseline_begin_offset, baseline_end_offset))
    baselined = baselined + 10
    resonance.createOutput(baselined, 'out')

    #как делать мат операции?

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

    cmd1 = resonance.pipe.filter_event(events, rule1)
    resonance.createOutput(cmd1, 'Evtout1')


    cmd2 = resonance.pipe.filter_event(events, rule2)

    eeg_windowized = resonance.cross.windowize_by_events(eeg_filtered, events, window_size, window_shift)
    baselined = resonance.pipe.baseline(eeg_windowized, range(baseline_begin_offset, baseline_end_offset))
    result_ = resonance.pipe.transform_to_event(baselined, makeEvent)
    resonance.createOutput(result_, 'Evtout2')

    cmd3 = resonance.pipe.filter_event(events, lambda evt: not(rule1(evt) or rule2(evt)))
    resonance.createOutput(cmd3, 'Evtout3')