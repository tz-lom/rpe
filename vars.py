window_size = 200
window_shift = -200  # -200
baseline_begin_offset = 0
baseline_end_offset = 100
thresholdForEEGInVolts = 0.0005  # 0.0005  # пороговая амплитуда помехи для оценки кандидата на нецелевую эпоху 500мкВ
thresholdForEMGInVolts = 0.1  # 0.05  # пороговая амплитуда ЭМГ для оценки кандидата на целевую эпоху
intervalEvtToEvtInSec = 2  # интервал между кандидатами на целевую/нецелевую эпоху, сек