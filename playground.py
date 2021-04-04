import numpy as np

import resonance.run
import resonance.pipe
import resonance.cross
import resonance.db
import process
from process import online_processing_old


def interleave_blocks(blocks):
    return sorted(blocks, key=lambda x: x.TS[0])


def artificial_eeg(sampling_rate, seconds, freq):
    si = resonance.si.Channels(1, sampling_rate)
    data = [
        resonance.db.Channels(si, time*1e9, np.sin(np.linspace(time, time+1, num=sampling_rate)*2*np.pi*freq))
        for time in np.arange(0, seconds)
    ]
    return si, data
# ф-ция создаёт сигналы-синусоиды указанной частоты для указанного кол-ва каналов
def artificial_eeg2(sampling_rate, seconds, freq, chCnt = 1):
    si = resonance.si.Channels(chCnt, sampling_rate)
    data = []
    for time in np.arange(0, seconds):
        data_ = np.zeros((sampling_rate, chCnt))
        for ic in range(chCnt):
            data_[:,ic] = np.sin(np.linspace(time, time+1, num=sampling_rate)*2*np.pi*freq[ic])
        resData = resonance.db.Channels(si, time*1e9, data_)
        data.append(resData)
    return si, data

freq = [4.3, 10, 15]
eeg_si, eeg_blocks = artificial_eeg2(500, 8, freq, 3)

#eeg_si, eeg_blocks = artificial_eeg(500, 8, 4.3)

events_si = resonance.si.Event()
events_blocks = [
    resonance.db.Event(events_si, 1.1e9, '1'),
    resonance.db.Event(events_si, 5.4e9, '2')
]

#events_blocks = []

si = [eeg_si, events_si]# Для чего нужны eeg_si и events_si?  si - stream info
data = interleave_blocks(eeg_blocks + events_blocks)


r1 = resonance.run.offline(si, data, process.online_processing)# что такое r1
r2 = resonance.run.online(si, data, process.online_processing_old)# что такое r2
print(r1['out'])
#print(r2['out'])