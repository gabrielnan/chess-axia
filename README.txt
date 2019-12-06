CSE 6242 Team 44 Final Project: Dynamic Chess Valuation with Deep Neural Networks

DESCRIPTION
This folder contains the deliverables for the Team 44 project, which include code for neural network model training and interactive visualization, and the poster and final report. 

The package contains an interactive chess tool designed to dynamically display relative chess values based on any board configuration. The UI is a chessboard the user can customize to test and play different scenarios/board configurations. Bar graphs outside the board will shift in tandem with piece values as plays are made. The player may either play from a standard chessboard starting position or free-form, placing pieces onto a blank board in any desired configuration. The tool also contains buttons for clearing and resetting the board as necessary.

The linked model that predicts piece value based on board configuration is a deep neural network linked to the UI. This model takes parsed board configurations from the original dataset and runs them through an auto-encoder, then runs the auto-encoder output through several neural networks (one for each chess piece type) and outputs an aggregated “centipawn” score that indicates a player’s advantage. 

INSTALLATION
This code uses Python 3.7.  To install the packages required for the visualization, complete the following steps:
1) Navigate to CODE/Visualization
2) Run pip install -r requirements.txt

EXECUTION
To execute the visualization, complete the following steps:
1) Navigate to CODE/Visualization
2) Run python app.py to serve the visualization to localhost:5001.  If another port is desired, provide the port number as an argument, like python app.py 6001
3) In a web browser, navigate to localhost:5001 (or specified port).  Initialize the board with the button 'Start Position.'  Then, play both sides of a chess game and see how the chess piece values change.
