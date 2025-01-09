from ai_system import ImageClassifier


def main():
    classifier = ImageClassifier(target_size=(224, 224))
    classifier.load_data()
    classifier.train_model(epochs=2)
    classifier.evaluate_model()

    sample_img = "sample_test_image.jpg"
    prediction = classifier.predict(sample_img)
    print(f"Predicted class index for {sample_img}: {prediction}")


if __name__ == "__main__":
    main()
