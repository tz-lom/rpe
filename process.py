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
    cut_off_frequency = 10  # change this to 3 to suppress signal
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')
    eeg_filtered = resonance.pipe.filter(eeg, low_pass_filter)
    resonance.createOutput(eeg_filtered, 'out')

def makeEvent(block):
    return np.max(block)
 # return block

def online_processing_old():
    eeg = resonance.input(0)

    cut_off_frequency = 30
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')

    eeg_filtered = resonance.pipe.filter(eeg, low_pass_filter)

  #  resonance.createOutput(eeg_filtered, 'out')

    events = resonance.input(1)

    window_size = 500
    window_shift = -250
    eeg_windowized = resonance.cross.windowize_by_events(eeg_filtered, events, window_size, window_shift)

    baseline_begin_offset = 0
    baseline_end_offset = 250
    baselined = resonance.pipe.baseline(eeg_windowized, baseline_begin_offset, baseline_end_offset)

    result = resonance.pipe.transform_to_event(baselined, makeEvent)
    resonance.createOutput(result, 'Evtout')

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

    events = resonance.input(1)

    window_size = 250
    eeg_windowized = resonance.cross.windowize_by_events(eeg_filtered, events, window_size)

    baseline_begin_offset = 0
    baseline_end_offset = 250
    baselined = resonance.pipe.baseline(eeg_windowized, baseline_begin_offset, baseline_end_offset)
    baselined = baselined + 10
    resonance.createOutput(baselined, 'out')
    #как делать мат операции?
    '''
    baselined = baselined + 10
    if len(events) > 0:
        resonance.createOutput(baselined, 'out')
    else:
        resonance.createOutput(eeg_filtered, 'out')
    '''

def online_processing():
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

    cmd1 = np.array(["1"])
    cmd2 = np.array(["2"])

    window_size = 500
    window_shift = -250
    baseline_begin_offset = 0
    baseline_end_offset = 250

    if np.array_equal(events, cmd1):
        pass
    elif np.array_equal(events, cmd2):
        eeg_windowized = resonance.cross.windowize_by_events(eeg_filtered, events, window_size, window_shift)
        baselined = resonance.pipe.baseline(eeg_windowized, baseline_begin_offset, baseline_end_offset)
        result_ = resonance.pipe.transform_to_event(baselined, makeEvent)
        resonance.createOutput(result_, 'Evtout')
    else:
        si = resonance.si.Event()
        result_ = resonance.db.make_empty(si)
        resonance.createOutput(result_, 'Evtout')