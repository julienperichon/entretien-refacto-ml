import gradio as gr
from ai_system import ImageClassifier

# Creating and training model at import time for the Gradio demo
global_classifier = ImageClassifier(target_size=(224, 224))
global_classifier.load_data()
global_classifier.train_model()


def classify_image(image):
    pred = global_classifier.predict(image)
    return str(pred)


def launch():
    demo = gr.Interface(
        fn=classify_image,
        inputs="image",
        outputs="text",
        title="Image Classifier",
        description="A small image classification example",
    )
    demo.launch()


if __name__ == "__main__":
    launch()
