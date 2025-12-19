.. PulseRetrievalwithJAX documentation master file, created by
   sphinx-quickstart on Fri Dec  5 17:41:21 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PulseRetrievalwithJAX
===================================

Overview
-------------------------


| This library aims to implement a variety of retrieval algorithms for pulse characterization methods. The available pulse characterization methods so far are Frequency-Resolved-Optical-Gating (FROG), Time-Domain-Ptychography (TDP), Two-Dimensional Spectral-Shearing Interferometry (2D-SI), Very advanced method for phase and intensity retrieval of E-fields (VAMPIRE) and Chirp-Scans based on material dispersion or pulse shaping. For all these methods second-harmonic generation (SHG), third-harmonic generation (THG), polarization gating (PG) and self-diffraction (SD) are supported. In addition, all methods (except Chirp-Scans) support cross-correlation and doubleblind retrieval.
| All implemented algorithms support the incorporation of additional constraints in the form of a pulse spectrum. Some will use these to perform a true constrained optimization, while others will project the current guess onto the available spectrum.

| The implemented algorithms may be split into two categories. Namely "classical" and "general" optimizers, where the term classical refers to algorithms, which have been specifically developed and applied in the context of pulse-retrieval. Some of these algorithms have been augmented through standard optimiztion approaches such as linesearch, newton-like methods and momentum.
| The term general refers to general optimization algorihms such as Gradient Descent or Differential Evolution. Since such algorithms are widely used and their implementations are readily available, the preexisting packages `Optax <https://github.com/google-deepmind/optax>`_, `Optimistix <https://github.com/patrick-kidger/optimistix>`_ and `evosax <https://github.com/RobertTLange/evosax>`_ are being used in this library.
| People who consider using this library should definitely look at `pypret <https://github.com/ncgeib/pypret>`_ the original implementation of the Common-Pulse-Retrieval-Algorithm (COPRA).




Content
========

.. toctree::
   :maxdepth: 1
   :caption: Modules:

   frog
   chirp_scan
   tdp
   twodsi
   vampire
   simulate_trace
   real_fields
   core
   utilities



.. toctree::
   :maxdepth: 1
   :caption: Equations:

   Definitions_and_Formulas

