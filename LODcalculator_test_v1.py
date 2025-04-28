import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt

data_path = r"C:\Users\tp-tanjk\OneDrive\PORCHE\1. Testing\1. LOD\Raw data chromos set up dark room uncovered fresh fluoro batch\TEX_240425_3.csv"

#folder_path = os.path.realpath(os.path.dirname(__file__))
#file_path = str(input("Enter the file name: "))
#data_path = os.path.join(folder_path, file_path)

data_df = pd.read_csv(data_path)
sensorcolour = ["HEX", "FAM", "TEX", "Cy5"]

available_substances = list(set(data_df['substance']))
# estimate end dataframe size
total_substances = len(available_substances)
total_concentrations = len(list(set(data_df['concentration'])))
total_leds = len(list(set(data_df['led'])))
total_rows = total_substances * total_leds * total_concentrations

summary_columns = ["substance", "concentration", "led"]
 
for substance in sensorcolour:
    summary_columns.append(f"{substance}_avg")
    summary_columns.append(f"{substance}_std")

lod_columns = ["substance", "led"]
for substance in sensorcolour:
    lod_columns.append(f"{substance}_lod")
    lod_columns.append(f"{substance}_std3")
    lod_columns.append(f"{substance}_slope")
    lod_columns.append(f"{substance}_r2")
    lod_columns.append(f"{substance}_intcp")




summary_df = pd.DataFrame(columns=summary_columns,
                          index=range(total_leds * total_substances * total_concentrations)
                          )
summary_index = 0

lod_df = pd.DataFrame(columns=lod_columns,
                      index=range(total_substances * total_leds)
                      )
lod_index = 0


@dataclass
class LODOutput:
    std3: float
    slope: float
    intercept: float
    lod: float
    r2: float


def calculate_lod(_df: pd.DataFrame, _led: str, _subs: str, _concs: list[int], sensor_num: str):

    # append to lod dataframe containing LOd results
    std3 = float(
        _df.loc[
            (_df["concentration"] == 0) &
            (_df["substance"] == _subs) &
            (_df["led"] == _led),
            f"{sensor_num}_std"]) * 3.3
    # get slope values
    slope_values = list()
    conc_values = list()
    for i_conc in _concs:
        slope_values.append(
            float(
                summary_df.loc[
                    (summary_df["concentration"] == i_conc) &
                    (summary_df["substance"] == _subs) &
                    (summary_df["led"] == _led),
                    f"{sensor_num}_avg"])
        )
        conc_values.append(i_conc)
    coefficients = np.polyfit(conc_values, slope_values, 1)
    slope, intercept = coefficients
    y_pred = np.polyval(coefficients, conc_values)
    ss_residual = np.sum((slope_values - y_pred) ** 2)  # Sum of squared residuals
    ss_total = np.sum((slope_values - np.mean(slope_values)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_residual / ss_total)
    lod = std3 / slope

    return LODOutput(std3, slope, intercept, lod, r_squared)


def plot_with_regression(x, y, sensor_name, output_path):
    """
    Plots a scatter plot with a linear regression line, equation, and R² value.

    Args:
        x (list or np.array): X-axis values (e.g., concentrations).
        y (list or np.array): Y-axis values (e.g., average sensor readings).
        sensor_name (str): Name of the sensor (e.g., "HEX").
        output_path (str): Path to save the plot image.
    """
    # Perform linear regression
    coefficients = np.polyfit(x, y, 1)
    slope, intercept = coefficients
    y_pred = np.polyval(coefficients, x)
    ss_residual = np.sum((y - y_pred) ** 2)  # Sum of squared residuals
    ss_total = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_residual / ss_total)
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data Points')
    plt.plot(x, y_pred, color='red', label=f'Fit Line: y = {slope:.6f}x + {intercept:.4f}')
    plt.title(f'{sensor_name} - Linear Regression')
    plt.xlabel('Concentration')
    plt.ylabel(f'{sensor_name} Average')
    plt.legend()

    # Annotate with the equation and R² value
    plt.text(0.05, 0.95, f'$y = {slope:.6f}x + {intercept:.4f}$\n$R^2 = {r_squared:.4f}$',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    # Save the plot
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
# analyze one substance at a time

for substance in available_substances:
    substance_df = data_df[data_df["substance"] == substance]
    concentrations = [int(i) for i in list(set(substance_df['concentration']))]
    leds = list(set(substance_df['led']))
    if len(concentrations) < 3:
        print(f"Only {len(concentrations)} concentrations for {substance}. Skipping.")
        continue
    if 0 not in concentrations:
        print(f"No 0 nM concentration for {substance}. Skipping.")
        continue
    for led in leds:
        led_df = substance_df[substance_df["led"] == led]
        for concentration in concentrations:
            # append to summary dataframe containing mean and std values
            concentration_df = led_df[led_df["concentration"] == concentration]
            summary_df.loc[summary_index, "substance"] = substance
            summary_df.loc[summary_index, "concentration"] = int(concentration)
            summary_df.loc[summary_index, "led"] = led
            for i, i_sensor in enumerate(sensorcolour):
                summary_df.loc[summary_index, f"{i_sensor}_avg"] = concentration_df.iloc[: , i + 3].mean()
                summary_df.loc[summary_index, f"{i_sensor}_std"] = concentration_df.iloc[: , i + 3].std()
            # summary_df.loc[summary_index, "green_avg"] = concentration_df.iloc[: , 3].mean()
            # summary_df.loc[summary_index, "green_std"] = concentration_df.iloc[: , 3].std()
            # summary_df.loc[summary_index, "blue_avg"] = concentration_df.iloc[: , 4].mean()
            # summary_df.loc[summary_index, "blue_std"] = concentration_df.iloc[: , 4].std()
            # summary_df.loc[summary_index, "lime_avg"] = concentration_df.iloc[: , 5].mean()
            # summary_df.loc[summary_index, "lime_std"] = concentration_df.iloc[: , 5].std()
            # summary_df.loc[summary_index, "red_avg"] = concentration_df.iloc[: , 6].mean()
            # summary_df.loc[summary_index, "red_std"] = concentration_df.iloc[: , 6].std()
            summary_index += 1

        # append to lod dataframe containing LOd results
        lod_df.loc[lod_index, "substance"] = substance
        lod_df.loc[lod_index, "led"] = led
        for i_sensor in sensorcolour:
            lod_result = calculate_lod(summary_df, led, substance, concentrations, i_sensor)
            #print(lod_result)
            lod_df.loc[lod_index, f"{i_sensor}_std3"] = lod_result.std3
            lod_df.loc[lod_index, f"{i_sensor}_slope"] = lod_result.slope
            lod_df.loc[lod_index, f"{i_sensor}_lod"] = lod_result.lod
            lod_df.loc[lod_index, f"{i_sensor}_r2"] = lod_result.r2
            lod_df.loc[lod_index, f"{i_sensor}_intcp"] = lod_result.intercept
            
        #
        # lod_df.loc[lod_index, "substance"] = substance
        # lod_df.loc[lod_index, "led"] = led
        # lod_df.loc[lod_index, "green_std3"] = float(
        #     summary_df.loc[
        #         (summary_df["concentration"] == 0) &
        #         (summary_df["substance"] == substance) &
        #         (summary_df["led"] == led),
        #         "green_std"]) * 3.3
        # # get slope values
        # slope_values = list()
        # conc_values = list()
        # for _conc in concentrations:
        #     slope_values.append(
        #         float(
        #             summary_df.loc[
        #                 (summary_df["concentration"] == _conc) &
        #                 (summary_df["substance"] == substance) &
        #                 (summary_df["led"] == led),
        #                 "green_avg"])
        #     )
        #     conc_values.append(_conc)
        # coefficients = np.polyfit(conc_values, slope_values, 1)
        # slope, intercept = coefficients
        # y_pred = np.polyval(coefficients, conc_values)
        # ss_residual = np.sum((slope_values - y_pred) ** 2)  # Sum of squared residuals
        # ss_total = np.sum((slope_values - np.mean(slope_values)) ** 2)  # Total sum of squares
        # r_squared = 1 - (ss_residual / ss_total)
        #
        # lod_df.loc[lod_index, "green_slope"] = slope
        # lod_df.loc[lod_index, "green_r2"] = r_squared
        # lod_df.loc[lod_index, "green_lod"] = float(lod_df.loc[lod_index, "green_std3"]) / slope
        lod_index += 1

summary_df = summary_df.fillna(0.0)

led_pairing = {"FAM" : "Blue" , "HEX" : "Green", "TEX" : "Lime", "Cy5" : "Red"}

key_df =  pd.DataFrame()
for k in range(lod_df.shape[0]):
    if lod_df.loc[k,"led"] == led_pairing[lod_df.loc[k,"substance"]]:
        fluorophore = lod_df.loc[k,"substance"]
        key_df.loc[k,"Fluorophore"] = lod_df.loc[k,"substance"]
        key_df.loc[k,"led"] = lod_df.loc[k,"led"]
        key_df.loc[k,f"{fluorophore}_lod"] = lod_df.loc[k,f"{fluorophore}_lod"]
        key_df.loc[k,f"{fluorophore}_std3"] = lod_df.loc[k,f"{fluorophore}_std3"]
        key_df.loc[k,f"{fluorophore}_slope"] = lod_df.loc[k,f"{fluorophore}_slope"]
        key_df.loc[k,f"{fluorophore}_r2"] = lod_df.loc[k,f"{fluorophore}_r2"]
        key_df.loc[k,f"{fluorophore}_intcp"] = lod_df.loc[k,f"{fluorophore}_intcp"]


# get folder name   
#grandparent_path = os.path.dirname(os.path.dirname(data_path))
folder_name = os.path.dirname(data_path)
base_name = os.path.splitext(os.path.basename(data_path))[0]
summary_df.to_csv(os.path.join(folder_name, base_name + '_summary.csv'))
lod_df.to_csv(os.path.join(folder_name, base_name + '_lod_crosstable.csv'))
key_df.to_csv(os.path.join(folder_name, base_name + '_key.csv'))


for substance in available_substances:
    # Filter rows where the "led" matches the "substance" according to led_pairing
    filtered_df = summary_df[
        summary_df["led"] == summary_df["substance"].map(led_pairing)
    ]

    # Extract x (concentration) and y (substance_avg) values
    x = filtered_df[filtered_df["substance"] == substance]["concentration"].values
    y = filtered_df[filtered_df["substance"] == substance][f"{substance}_avg"].values

    # Skip if there are not enough points to plot
    if len(x) < 2:
        print(f"Not enough data points for {substance}. Skipping plot.")
        continue

    # Define output path for the plot
    plot_output_path = os.path.join(folder_name, f"{base_name}_plot.png")

    # Generate the plot
    plot_with_regression(x, y, substance, plot_output_path)
    print(f"Plot saved for {substance} at {plot_output_path}")