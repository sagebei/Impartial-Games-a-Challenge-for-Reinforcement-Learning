
## Impartial Games - A Challenge for Reinforcement Learning

This repository contains all the python scripts used in the experiments of this paper, which are organized in the folders as following:

1. **example**: includes the codes and experiment setup for the example 2 in section *4. Different level of mastery*
2. **value networks**: includes the codes used in section *5.1 Model parity function using value networks*. 
3. **policy networks**:  includes the codes used in section *5.2 Learn winning moves using policy networks*
4. **reinforcement learning**: includes the codes used in section *7. Reinforcement learning on for nim*

### Notes on using our codes:
* We would recommend running our codes in a python virtual environment, which can be created by running 
```python -m venv venv```
in a terminal. Then download our repository and install all the required packages by
```pip install -r requirments.txt```. You now will be ready to run any script without being bothered by the error message that says "cannot find xxx packge". 
* While our implemtations natively support calcuating the Elo rating score for the agent being trained, it demands signicantly amount of calcuation. Thus we suggest disabling this functinality by setting ```'calculate_elo'``` in the ```args``` configuation variable to ```false``` if you does not consider the relative strong of the agent being trained in comparasion with its ancestors. 
* We use Ray library to enable running simulations in parallel. An error might raise if you set the ```num_workers``` more than the number of CPUs available on your machine. 
* To run the experiments and replicate our results, please refer to our paper for the experiment configurations. The trained models are stored in the **model** folder. The name of the model files reflexs the board size of the nim they were trained on. For example, the file named **5heaps** is the model for the 5 heaps nim. We support conducting analysis on specifed nim positions on 5, 6 and 7 heaps using trained models. The results on running the analysis on the initial position of a 5 heap nim are shown below. 

<img src="https://github.com/sagebei/Impartial-Games-a-Chanllenge-to-Reinforcement-Learning/blob/main/images/analysis_on_nim_board_position.png" alt="drawing" width="1300"/>


### Bibtex
```
Cite our paper by 
```


