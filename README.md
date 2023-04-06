# MOL3022 Project

### Installation
#### SSH
```
git clone git@github.com:mattiaseggen/mol3022.git
```

#### HTTPS
```
https://github.com/mattiaseggen/mol3022.git
```

Download the [dataset](https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure?select=2018-06-06-ss.cleaned.csv),
and place it into a folder named "data" in the project's root directory.

Install the required dependencies by running ```make init``` in the root directory.

### Running the program
To run the program, the model must be trained first. This is done by executing ```make train_model```.
A prediction can be made by executing ```python3 run.py [sequence]```.
