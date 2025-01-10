Brief code explanation:

- Color normalization
ChromaCalculate: calculate the color transform matrix between two doamains (different scanners).
ColorNormalize: use calculated transform matrix to carry out transformation on WSI file

- Data process
DataAggreation: convert feature data of Qupath program into an integrated .csv file 

- Machine learning
MLTrainNEval: train ML models over chosen dataset
MLTest: evaluate ml model on target dataset, calculate and record scores

- MLP learning
MLPTrain: train a simple MLP over chosen dataset
MLPTest: evaluate MLP model on target dataset, using Case model(case level celluar features), Patch model(patch level cellular features) and GFLOW(a blend model using both level features) to get predict results. Calculate scores and generate figures.

P.S. I do not filter extra packages in the "requirements.txt" ...... 
P.P.S. to fluently run the ColorNormalize function, a WSI file is needed. But, this kind of file is too big ... so please contact me when necessary. 
