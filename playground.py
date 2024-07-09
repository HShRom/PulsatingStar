import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
# from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

data = pd.read_excel(r"C:\Users\Arik Drori\Desktop\Year3+\Astro Exp\TYC 1514-827-1\alldata.xlsx")
relevant_headers = ["Time(Hours)",
                    "Source-Sky_T1", "Source_Error_T1",
                    "Source-Sky_C2", "Source_Error_C2",
                    "Source-Sky_C3", "Source_Error_C3",
                    "Source-Sky_C4", "Source_Error_C4",
                    "Source-Sky_C5", "Source_Error_C5"]

my_data = data[relevant_headers]
print(my_data)

middle_plot = True

if middle_plot:
    plt.xlabel('Time (Hours)')
    plt.ylabel('Counts')
    plt.title('Photon counts of stars in relation to Time (Hours)')
    labels = ["Origin Star T1", "Comparison Star C2", "Comparison Star C3", "Comparison Star C4", "Comparison Star C5"]
    for i in range(5):
        plt.errorbar(x=my_data[relevant_headers[0]], y=my_data[relevant_headers[1+2*i]], yerr=my_data[relevant_headers[2+2*i]], fmt='.', label=labels[i])
    plt.legend()
    plt.show()

avg_comparison_data = np.mean(data[[relevant_headers[1],relevant_headers[3],relevant_headers[5],relevant_headers[7]]],axis = 1) ## b
avg_comparison_err = 0
for i in range(4):
    avg_comparison_err += np.power(data[relevant_headers[2+2*i]],2)
avg_comparison_err = np.sqrt(avg_comparison_err) ## error b

star_data = np.array(my_data[relevant_headers[1]])  # a
star_error = np.array(my_data[relevant_headers[2]])  # error_a


division_error = star_data/avg_comparison_data * np.sqrt(
    np.power(star_error/star_data,2) + np.power(avg_comparison_err/avg_comparison_data,2)
)

division = star_data/avg_comparison_data

log_error_array = -2.5*(division_error/division)
time_array = my_data[relevant_headers[0]]
magnitude_array = -2.5*np.log(division)
N = len(time_array)

raw_data_plot = False
if raw_data_plot:
    plt.xlabel('Time (Hours)')
    plt.ylabel('Magnitude Difference (Delta M)')
    plt.title('Delta M of T1 in relation to Time (Hours)')
    plt.errorbar(x=time_array, y=magnitude_array, yerr=log_error_array, fmt='.', label= "Normalized Star Magnitude Differenece")
    plt.legend()
    plt.show()



## Get f_0 from LS
show_power_spectrum = True
if show_power_spectrum:
    frequencies = np.linspace(1/(10000*N),10,N*1000)
    periodogram = lombscargle(time_array, magnitude_array,frequencies)
    plt.xlabel('Frequency (1/Hours)')
    plt.ylabel('Power')
    plt.title('Lombscargle Power Spectrum [Power to Frequency]')
    plt.plot(frequencies/(2*np.pi), periodogram)
    plt.show()

f_0 = 0.71
w_0 = 2*np.pi*f_0


## Fit fourier series of third order





# Define the chi-squared cost function

def fit_and_plot_fourier_and_residuals(omega):

    def third_order_fourier_series_func(x, b0, a1, b1, a2, b2, a3, b3, w=omega):
        return b0 + a1 * np.sin(w * x) + b1 * np.cos(w * x) + a2 * np.sin(2 * w * x) + b2 * np.cos(
            2 * w * x) + a3 * np.sin(3 * w * x) + b3 * np.cos(3 * w * x)

    # Initial guess for the parameters (adjust as needed)
    p0 = [1, 1, 1, 1, 1, 1, 1]

    # Perform the fit using curve_fit
    popt, pcov = curve_fit(third_order_fourier_series_func, time_array, magnitude_array, p0=p0, method='lm')

    # Print the optimal parameters
    print("Optimal parameters:", popt)

    # Calculate the fitted function with the optimized parameters
    y_fit = third_order_fourier_series_func(time_array, *popt)
    plt.errorbar(x=time_array, y=magnitude_array, yerr=log_error_array, fmt='.', label= "Data - Delta M")
    plt.plot(time_array, y_fit, label='Fourier Series Fit')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Delta M')
    plt.title('Third Degree Fourier Series Fit with f_0 = 0.75, With Data')
    plt.legend()
    plt.show()

    plt.errorbar(x=time_array, y=y_fit-magnitude_array, yerr=log_error_array, fmt='.', label="Delta M - F(t_i)")
    plt.xlabel('Time (Hours)')
    plt.ylabel('Delta M - F(t_i)')
    plt.title('Residual graph of Delta M Data to Third Degree Fourier Series Fit with f_0 = 0.75')
    plt.show()
    return popt, third_order_fourier_series_func


optimized_parameters, fourier_series_fit = fit_and_plot_fourier_and_residuals(w_0)

calculate_optimal_w = True
if calculate_optimal_w:
    def chi_squared(x, y, y_err, b0, a1, b1, a2, b2, a3, b3, w):
        f_fit = fourier_series_fit(x, b0, a1, b1, a2, b2, a3, b3, w)
        return np.sum(((y - f_fit) / y_err) ** 2)

    N = 10000
    w_arrays = np.linspace(w_0-0.13,w_0+0.13,N)
    chi_squared_arr = np.zeros(N)
    for i in range(N):
        chi_squared_arr[i] = chi_squared(time_array, magnitude_array, log_error_array, *optimized_parameters, w_arrays[i])

    plt.xlabel('w_0 (Rad/Hour)')
    plt.ylabel('Chi Squared')
    plt.title('Chi Squared in relation to w in small range over w_0')
    plt.plot(w_arrays, chi_squared_arr)

    min_ind = np.argmin(chi_squared_arr)
    minimal_chi_squared = chi_squared_arr[min_ind]
    optimal_w = w_arrays[min_ind]
    plt.axhline(y=minimal_chi_squared, color='red', linestyle='--', label=f"minimal chi squared: {minimal_chi_squared:.1f}")
    plt.axhline(y=minimal_chi_squared+1, color='green', linestyle='--', label=f" minimal chi squared + 1: {minimal_chi_squared+1:.1f}")

    plt.axvline(x=optimal_w, color='black', linestyle='--', label=f"w value for minimal chi squared: {optimal_w:.3f} rad/Hour")
    plt.axvline(x=4.71856, color='orange', linestyle='--', label=f"w value for lower end chi squared + 1 difference: 4.718 rad/Hour")
    plt.axvline(x=4.73475, color='blue', linestyle='--', label=f"w value for upper end chi squared + 1 difference: 4.734 rad/Hour")

    plt.legend()

    plt.show()

    print(optimal_w/(2*(np.pi)))
    print((optimal_w-4.71856)/(2*np.pi))
    #print(4.73475 - optimal_w)
    print("Optimal Period: " + str((1/12)*np.pi/optimal_w) + " days.")
    optimized_parameters, fourier_series_fit = fit_and_plot_fourier_and_residuals(optimal_w)



def plot_segmented_arrays(x, y, c, low_y_const = 0.7):
  """
  Plots arrays x and y in segments based on the dynamic value c.

  Args:
      x: Array of x-coordinates.
      y: Array of y-coordinates (same size as x).
      c: Dynamic value defining segment boundaries.
  """
  num_segments = int(np.ceil(x[-1] / c))  # Calculate number of segments
  start = 0
  end = min(x[-1], c)  # Initialize starting and ending points

  y_diff = np.max(y) - np.min(y)

  for segment in range(num_segments):
    # Plot the current segment
    values_to_print = np.bitwise_and(x > start,x < end)
    plt.scatter(x[values_to_print] - start, y[values_to_print] -low_y_const*y_diff*segment, s=200, c=y[values_to_print], cmap='Oranges')
    #plt.plot(, , 'o', color='b')

    # Update starting and ending points for the next segment
    start = end
    end = start+c

  plt.xlabel("Time (Hours) Modulu C")
  plt.ylabel("Delta M Plus yDiff_i")
  plt.title("Synchronization view of Time Of Arrival / Delta M")
  # plt.show()

# Example usage (replace with your actual arrays)


# time_array = my_data[relevant_headers[0]]
# magnitude_array = -2.5*np.log(division)
#plot_segmented_arrays(np.array(time_array), np.array(magnitude_array), 1.1)

# Function to update plot based on new c value
def update_plot():
  global c_value, d_value  # Access global variables

  try:
    new_c = float(entry_c.get())  # Get new c value from entry field
    c_value = new_c  # Update global c value

    new_d = float(entry_d.get())  # Get new d value from entry field
    d_value = new_d  # Update global d value

    # Generate plot with new c value
    ax.clear()
    plot_segmented_arrays(np.array(time_array), np.array(magnitude_array), c_value)

    # Redraw canvas to update plot
    canvas.draw()

  except ValueError:
    # Handle potential errors (e.g., non-numeric input)
    pass


def handle_key_press(event):
  global c_value, d_value

  # Get the pressed key
  key = event.keysym

  # Update c value based on arrow key presses
  if key == 'Right':
    c_value += d_value
  elif key == 'Left':
    c_value -= d_value

  # Ensure c stays positive
  c_value = max(c_value, 0)  # Avoid negative c values

  # Update plot with new c value
  update_plot()


c_value = 1
d_value = 0.001
# Create main window
root = tk.Tk()
root.title("Segmented Plot with Dynamic Control")

# Create Matplotlib figure and axis beforehand
fig, ax = plt.subplots()

# Create plot canvas
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Label and entry for c value
label_c = tk.Label(root, text="Dynamic c value:")
label_c.pack(side=tk.LEFT)

entry_c = tk.Entry(root, width=10)
entry_c.insert(0, str(c_value))  # Pre-fill with initial c value
entry_c.pack(side=tk.LEFT)

# Label and entry for d value
label_d = tk.Label(root, text="Step size (d):")
label_d.pack(side=tk.LEFT)

entry_d = tk.Entry(root, width=5)
entry_d.insert(0, str(d_value))  # Pre-fill with initial d value
entry_d.pack(side=tk.LEFT)

# Button to update plot
button_update = tk.Button(root, text="Update Plot", command=update_plot)
button_update.pack(side=tk.LEFT)

# Bind arrow key press event
root.bind("<KeyPress>", handle_key_press)

# Initial plot
plot_segmented_arrays(np.array(time_array), np.array(magnitude_array), c_value)

# Run the main loop
root.mainloop()



