from typing import Dict
import torch
from transformers import pipeline
from pprint import pprint


# Load the zero-shot classification model
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/ModernBERT-large-zeroshot-v2.0",
    torch_dtype=torch.bfloat16,
    device=-1
)


def classify(
    text_input: str, labels: str, multi_label: bool = False
) -> Dict[str, float]:
    """
    Performs zero-shot classification on the given text input and candidate labels.

    Args:
        - text_input: The input text to classify.
        - labels: A comma-separated string of candidate labels.
        - multi_label: A boolean indicating whether to allow the model to choose multiple classes.

    Returns:
        Dictionary containing label-score pairs.
    """

    # Perform zero-shot classification
    hypothesis_template = "This text is about {}"
    prediction = classifier(
        text_input,
        labels,
        hypothesis_template=hypothesis_template,
        multi_label=multi_label,
    )

    return {
        prediction["labels"][i]: prediction["scores"][i]
        for i in range(len(prediction["labels"]))
    }


text = "Wolf sees a car. What will the wolf do?"
pprint(classify(text_input=text, labels="cat,wolf,car,something else"))
