# Fourier Based Approximation with Thresholding (FBAT)

It is a method that allows to classify time series. 

## How to install

Install all requirements from pip:
```
pip install -r requirements/requirements.out
```
## How to use
You can start by using the main file that contain an example of use:
```
python main.py --fileTrain <path of train dataset> --fileTest <path of test dataset> --fileOut <path of results file>
```
Each dataset is composed of time series and labels. Each line corresponds to a time series and the first character corresponds to the label of the time series. Each character must be separated by a space.
