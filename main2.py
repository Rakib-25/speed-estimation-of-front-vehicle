import numpy as np
import matplotlib.pyplot as plt

# Example x and y values of the curve
x_values = np.array([319, 174, 98, 79, 62, 60, 51, 45])  # Example x values
y_values = np.array([0.15619387721729117, 0.17452354936446288, 0.1991195147817372, 0.20129095510651626, 0.19011922672684847, 0.2147238996051384, 0.2089093441094998, 0.2075858274570303])  # Corresponding y values

# Perform polynomial interpolation
degree = 4  # Degree of the polynomial (quadratic curve)
coefficients = np.polyfit(x_values, y_values, degree)
curve_function = np.poly1d(coefficients)

# Test the curve function with some x values
test_x_values = np.array([50,100,150,200,250,300,350])
predicted_y_values = curve_function(test_x_values)

# Plot the original curve and the interpolated curve
plt.scatter(x_values, y_values, label='Original curve', color='blue')
plt.plot(test_x_values, predicted_y_values, label='Interpolated curve', color='red')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Original and Interpolated Curves')
plt.legend()
plt.grid(True)
plt.show()
