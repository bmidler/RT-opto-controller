"""Real-time closed-loop optogenetics controller.

Fuses campy (FLIR camera acquisition + video writing) with the RT-opto
CNN-GRU behaviour classifier to drive a laser via a stim Arduino, in real
time, at a fixed frame rate.
"""
