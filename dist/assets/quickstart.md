# Quick Start #1 - Using Scikit-learn
In this tutorial we show how to deploy and run and end-to-end model on Shipped Brain's platform:
1. Train a simple linear regression model using `scikit-learn` and save it
2. Package the code using Shipped Brain project template
3. Deploy the Shipped Brain project with the pre-trained model to the platform
4. Make predictions using the model's unique API endpoint, automatically created by Shipped Brain during deployment

<br></br>
## <span class="color-magenta">What you need to get started</span>
To get started you need:
- Python >=3.6
- Create or clone a new project directory

**Options:**

- Clone this project and run it locally _recommended_
- Clone Shipped Brain Project Template _link_ and follow this tutorial step-by-step
    
The project template is the easiest way to start building your own end-to-end models with Shipped Brain. You don't need to start from scratch. The template contains all the files used by Shipped Brain to run and build your project. The files are pre-filled with a basic example that your should edit to meet your model's requirements.
These files include: 
- a template `config` file that you should fill with information about your model: _model name_, _pre-trained model file_, relative path to your python `requirements` file, etc.
- an `input_example` file in csv format with an illustrative input example that your model accepts
- an empty `requirements.txt` file that's mapped in `config` where you should include your project's required python packages
- a `model.py` file with a basic model class that interfaces with Shipped Brain - also mapped in `config`


## Getting Started
Once you've have create a project directory or cloned one of the repositories you can build your end-to-end model using Shipped Brain. 

Now, lets create a new python environment in our project's root directory.

First, change to the project's directory:

```bash
cd /path/to/project/dir
```

Create python environment inside project:

```bash
python -m venv ./venv
```
**NB:** make sure you have `python-venv` installed.

Activate the environment:

```bash
. ./venv/bin/activate
```

### Installing the packages
Before diving into our model we're first going to install our requirements:
- `numpy`: to manipulate our data
- `pandas`: to read and format our data
- `scikit-learn`: to create our model
- `joblib`: to store our model

```bash
pip install numpy pandas scikit-learn joblib
```

## Training the model 
We're going to train a linear regression model on the `wine_quality.csv` dataset using `sklearn` and save the trained model to the local directory using `joblib`.

To train and save a model you just need to run the `train.py` file:
```python
# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import joblib

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    np.random.seed(46) # Go Valentino Rossi

    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Get model's params.
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("\tRMSE: %s" % rmse)
    print("\tMAE: %s" % mae)
    print("\tR2: %s" % r2)

    # Save the model
    joblib.dump(lr, './model.pkl)
```

To run the example just run the command: `python train.py`

If you want to run the model with custom arguments you can also try: `python train.py <alpha> <l1>`

## Getting things ready for deployment - Packaging the Code :package: :brain:
Once the model has been trained and saved we can start packaging our code to deploy and get a custom endpoint on the Shipped Brain platform.

### The model class
For Shipped Brain to run your model you need to implement a class interface for your model. This class tells Shipped Brain how to load and run your model.
You can start with the `model.py` class from _template project_.

The `model.py` file:

```python
import joblib
import numpy as np
import pandas as pd
from typing import Union

class UserModel:

    def __init__(self, model_path: str) -> None:
        # load model from relative path
        self.model = joblib.load(model_path)

    def predict(self, input_features: pd.DataFrame) -> Union[np.ndarray, pd.Series, pd.DataFrame]:

        # Use scikit-learn predict method
        return self.model.predict(input_features)
```

As you can see you just need to load your model from the project's relative path and implement the `predict` method. Shipped Brain will use this class to do inference.

`IMPORTANT`: the pre-trained model file must be in the project directory, otherwise the model wont be able to run on the platform.

**NB**: the signature of the `predict` method tells that it accepts a `pd.DataFrame` as input and returns either a `pd.DataFrame` or an `np.ndarray` object. You should respect this, otherwise your model may not be able to run

### Showing users an input example
In Shipped Brain we value ease of use and integration above all else. When deploying a model you must also specify a file - `csv` or `json` - with a real input example that your model can draw predictions from. **If the input example isn't valid the deployment will fail**.

An input example - `input_example` file
```
"fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"
7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8
```

### :snake: Python requirements
Shipped Brain needs your python package dependencies in order to build your models. We're going to create a `requirements.txt` file with all our project's  python dependencies. To assure that the right packages are installed we provide the specific version of each package.

**NB:** not providing the correct package versions may lead to inconsistencies or even make your model impossible to run on the platform.

The `requirements.txt` file should look like this:
```
numpy
pandas
scikit-learn==0.23.2
joblib==0.16.0
```

You can either create the file by hand or run the command:

```bash
pip freeze > requirements.txt
```

### Project Config
Finally, we just need to set the blueprint of our project. Once we've implemented the class interface we need to tell Shipped Brain which files to load when doing inference.

For that we use the `config` file:

```yml
model_name: "ElasticNet-Quicky"             # you model's name on the platform
model_class: "UserModel"                    # the name of the model class
model_code: "model.py"                      # file with model_class implementation
code_path:                                  # no need for this, we don't have any code dependencies
input_example: "input_example"       # file with model input example
input_example_type: "csv"                   # the file type of the input example, either csv or json
requirements: "requirements.txt"            # the python requirements file

# Arguments passed to the model's class __init__
model_args:         
    model_path: "model.pkl"               # custom class argument
```

`IMPORTANT`: do not change the file name or path, the `config` file must be in the project's root


### Deployment :arrow_up:
Now we're ready to deploy our model and start predicting. 

To deploy the model we just need to go to the Shipped Brain platform: 
1. Login to your Shipped Brain account
2. Go to `Deploy Model`; on the top right of the screen
3. Select the `Upload file` button and select the zipped project file
4. Write a fancy model description; here we're just copy pasting the content of the `model_description.md` file
5. Click the `Upload` button

Congratulations! Your model has been Shipped.

### Making predictions
The model is now deployed to the platform and we can access it through its custom URL: `https://shipped.com/models/<model_name>` (replace with your own model's name).

You can make predictions directly on the platform using the in app json and `Try me` button.

You can also try it with `cURL`, just click on the json snippet to copy the command and paste it on the command line.

IMAGENS

## Custom models
Now that you know how develop end-to-end models you can start serving and sharing your own using Shipped Brain and let others try out your ML :heart: .

# That's all folks!