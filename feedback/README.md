## Classification confidence visualization

Displays confidence values for each class (left, right, headlights, rest).

Protocol: LSL
Input: signal with 4 channels (1: left, 2: right, 3: headlights, 4: rest)
Value range: 0-100
Update freq.: every 100ms

How to:
- Add and connect a "LSL Export (Gipsa)" box in the OpenViBE scenario
- In the box settings, set stream type to "SIG"
- Start the scenario
- Run visGUI.py

(Works for eg. with Sinus Oscillator box set to 4 channels)