import resonance
import resonance.pipe
import resonance.cross
import scipy.signal as sp_sig

def online_processing_example2():
    import resonance
    import resonance.pipe
    import scipy.signal as sp_sig
    eeg = resonance.input(0) 
    cut_off_frequency = 30  # change this to 3 to suppress signal
    low_pass_filter = sp_sig.butter(4, cut_off_frequency / eeg.SI.samplingRate * 2, btype='low')
    eeg_filtered = resonance.pipe.filter(eeg, low_pass_filter)
    resonance.createOutput(eeg_filtered, 'out')

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
    result = resonance.pipe.baseline(eeg_windowized, baseline_begin_offset, baseline_end_offset)

    resonance.createOutput(result, 'out')