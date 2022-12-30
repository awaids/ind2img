# ind2img - Indicator data to Image.

This module can be used to convert stock data into an image. any stock data comprising of the columns `Open`, `High`, `Low`, `Close` and `Volume` is used to generate the indicators data.

## Compute indicator data:
For a given stock data in pandas DataFrame, the `CumputeTALibIndicators` class uses [ta-lib](https://mrjbq7.github.io/ta-lib/) to compute indicator data and appends it to the DataFrame. For certain indicators, time period is required. Using the CumputeTALibIndicators, you can compute indicators data for multiple time periods.

## Transform to images/gif:
Using the helper functions in [array2img.py](array2img.py), you can convert the indicator data to images. You can either convert to BW images or to RGB images.

![](Data/bw0.png)
![](Data/rgb0.png)

Once you have the images, you can either save them to a dir or you can convert them to gifs and visualize them.

![](Data/bw.gif)
![](Data/rgb.gif)
