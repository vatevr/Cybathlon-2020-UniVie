This package contains scripts for analyzing eeg signals and extracting the average amplitude for specified frequency ranges for each electrode channel. Moreover it contains evaluation scripts.

There are three different implementations for this amplitude extraction, some of which require command line arguments:
 * ```amplitude_extraction.py``` (FFT) - **Cmd Line Arguments**: 1. Window-Size (eg 10.) 2. Sample rate (eg. 500)
 * ```amplitude_extraction_welch.py``` (Welch method) **Cmd Line Arguments**: 1. Sample rate (eg. 500)
 * ```amplitude_extraction_multitaper.py``` ()Multitaper) 
 
 It also contains an OpenVibe box with the same functionality that can be hooked into an OpenVibe pipeline, and uses the FFT algorithm. 
 
 There are three evaluation scripts for the three algorithms, which calculates the correlation coefficient for eeg data between two different subjects, and plots a heatmap. The algorithms are rewritten to not need any cmd line arguments, as it is just for evaluation.

Some of the scripts use a script provided by Anja Meunier for plotting topographies using mne plotting functions. This script contained in ```topomap_plot.py```. Parts of the data pre-processing are also from a template provided by her.

 The following datasets are used in the implementations/evaluations, and are not pushed on git, because they are personal data from the patients: ```20191210_Cybathlon_SAZ_Session1.*``` , ```20191201_Cybathlon_TF_Session1_Block1.*```, ```20191201_Cybathlon_TF_Session1_Block2.*```, ```20191201_Cybathlon_TF_Session1_RS.*``` , ```S4_4chns.raw```, ```S2_4chns.raw```