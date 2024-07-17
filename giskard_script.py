import giskard
from transformers import pipeline
import pandas as pd

# Load the model
model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define prediction function
def predict(text):
    result = model(text)[0]
    return {"label": result["label"], "score": result["score"]}

# Create a larger dataset for more comprehensive scanning
dataset = giskard.Dataset(
    pd.DataFrame({
        "text": [
            "I love this movie!", 
            "This film is terrible.", 
            "The acting was mediocre.", 
            "An absolute masterpiece!", 
            "I fell asleep during the movie.",
            "The plot was confusing and hard to follow.",
            "The special effects were amazing!",
            "I couldn't stop laughing throughout the film.",
            "The dialogue felt forced and unnatural.",
            "A disappointing sequel that didn't live up to the original."
        ]
    }),
    name="movie_reviews"
)

# Initialize Giskard model
giskard_model = giskard.Model(
    model=predict,
    model_type="classification",
    classification_labels=["POSITIVE", "NEGATIVE"],
    feature_names=["text"],
    name="sentiment_analysis"
)

# Run Giskard scan
scan_results = giskard.scan(giskard_model, dataset)

# Generate and save the report
scan_results.generate_report(output_path="giskard_report.html")

print("Giskard report has been generated and saved as 'giskard_report.html'")

# Print a summary of the scan results
print("\nScan Results Summary:")
print(f"Number of tests run: {len(scan_results.tests)}")
print(f"Number of tests passed: {sum(1 for test in scan_results.tests if test.passed)}")
print(f"Number of tests failed: {sum(1 for test in scan_results.tests if not test.passed)}")

# Print details of failed tests
failed_tests = [test for test in scan_results.tests if not test.passed]
if failed_tests:
    print("\nFailed Tests:")
    for test in failed_tests:
        print(f"- {test.name}: {test.description}")
else:
    print("\nAll tests passed!")

# Print top vulnerabilities (if any)
vulnerabilities = scan_results.vulnerability_ranking
if vulnerabilities:
    print("\nTop Vulnerabilities:")
    for vulnerability in vulnerabilities[:5]:  # Print top 5 vulnerabilities
        print(f"- {vulnerability.name}: {vulnerability.description}")
else:
    print("\nNo vulnerabilities detected.")