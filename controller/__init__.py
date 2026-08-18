"""Real-time closed-loop optogenetics controller.

Fuses FLIR camera acquisition (utils/flir.py, vendored from campy) with the
CNN-GRU behaviour classifier (utils/model.py) to drive a laser via a stim
Arduino, in real time, at a fixed frame rate. Video encoding, the preview
window, the camera-trigger serial protocol and the stim logic all live here.
"""
