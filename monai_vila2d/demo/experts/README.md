# Adding a new expert model

[BaseExpert](./base_expert.py) is a template for adding a new expert model to the repository. Please follow the instructions below to add a new expert model.

1. Create a new python file in the `monai_vila2d/demo/experts` directory. The file name should be the name of the expert model you are adding. For example, if you are adding a new expert model called `MyExpert`, the file name should be `my_expert.py`.

2. Implement the expert model `mentioned_by` method in the new python file. The method should return a boolean value indicating whether the expert model is mentioned by the given text from the language model. For example:

```python
def mentioned_by(text: str) -> bool:
    return "my expert" in text
```

3. Implement the expert model `run` method in the new python file. The method should return the expert model's prediction given the input image. For example:

```python

def run(image_url: str) -> Tuple[str, str, str, str]:
    # Load the image
    image = load_image(image_url)

    # Perform inference
    prediction = my_model(image)

    # Return the prediction
    text_output = "The expert model prediction is a segmentation mask."
    # save the prediction to a file
    save_image(prediction, "prediction.png")
    image_output = "prediction.png"
    instruction = "Use this mask to answer: what is the object in the image?"
    seg_file = "prediction.png"
    return text_output, image_output, instruction, seg_file
```
