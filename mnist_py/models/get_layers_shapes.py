from train import models_initializers

model_name = 'basic_nn'

model = models_initializers.get(model_name)()

for l in model.layers:
    print(f'{l.name} -- {l.input_shape} -- {l.output_shape}')
