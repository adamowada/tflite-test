# Tflite Test
Adam Owada

### About

Tflite deployment proof of concept on Vercel.

Turn a Keras model into a `.tflite` model like this:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model=your_model_name)
your_model_name_tflite = converter.convert()

with open("your_model_name.tflite", "wb") as file:
    file.write(your_model_name_tflite)

```


### Usage

Make a request like this:

```
https://tflite-test.vercel.app/api/regression?number=42
```

And you will receive the model's prediction as a response:

```
The model predicted: 51.96
```
