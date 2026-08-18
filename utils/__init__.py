"""Vendored third-party pieces the RT-opto controller depends on.

Everything here was extracted from the two upstream projects this controller
was built on top of; only the parts reachable from `python -m controller.run`
were kept.

    flir.py       <- campy (campy/cameras/flir.py) -- FLIR/PySpin acquisition,
                     used only when config `source: "flir"`.
    model.py      <- RT-opto (model.py) -- the CNN-GRU VideoClassifier that
                     controller/classifier.py loads the checkpoint into.
    checkpoints/  <- RT-opto (output/) -- trained weights.

Nothing else from either project is used at run time: video encoding, the
camera-trigger serial protocol, the live preview and the stim logic are all
implemented in controller/.
"""
