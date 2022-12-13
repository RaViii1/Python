from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
def add_inseption_module(input_tensor):
    act_func = 'relu'
    paths = [
    [Conv2D(filters = 96, kernel_size=(1,1),
    padding='same', activation=act_func),
    Conv2D(filters = 128, kernel_size=(1,1),
    padding='same', activation=act_func)
    ],
    [Conv2D(filters = 16, kernel_size=(1,1),
    padding='same', activation=act_func),
    Conv2D(filters = 32, kernel_size=(3,3),
    padding='same', activation=act_func)
    ],
    [MaxPooling2D(pool_size=(1,1),
     padding='same'),
    Conv2D(filters = 32, kernel_size=(1,1),
    padding='same', activation=act_func)
    ],
    [Conv2D(filters = 64, kernel_size=(1,1),
    padding='same', activation=act_func)
    ]
    ]
for_concat = []
for path in paths:
    output_tensor = input_tensor
    for layer in path:
        output_tensor = layer(output_tensor)
        for_concat.append(output_tensor)
    return concatenate(for_concat)
