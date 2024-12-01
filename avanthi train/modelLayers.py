from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import DenseNet121

model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Visualize the model and save the plot as a PNG
plot_model(model, to_file='densenet121_model.png', show_shapes=True, show_layer_names=True)
