import os
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import confusion_matrix  # type: ignore
from sklearn.metrics import ConfusionMatrixDisplay
from inference import predict_class
from prompts import evaluation_dataset_generation_prompt


def evaluate_classification_model_on_dataset(
    model_parameters: dict[str, str], val_dataset: list[str]
) -> tuple[list[str], list[str]]:
    """Returns the predictions and ground truths for the given dataset."""
    predictions, ground_truths = [], []
    with ThreadPoolExecutor() as executor:
        future_to_row = {
            executor.submit(
                predict_class, **model_parameters, verbatim=row.split("//")[0].strip()
            ): row
            for row in val_dataset
            if row and "//" in row
        }
        for future in tqdm(
            as_completed(future_to_row),
            total=len(val_dataset),
            desc="Evaluating model on dataset...",
        ):
            pred, label = (future.result().category, future_to_row[future].split("//")[1].strip().lower())  # type: ignore
            predictions.append(pred)  # type: ignore
            ground_truths.append(label)  # type: ignore
    return predictions, ground_truths  # type: ignore


def display_confusion_matrix(
    ground_truths: list[str], predictions: list[str], labels: list[str]
):
    cm = confusion_matrix(y_true=ground_truths, y_pred=predictions, labels=labels)  # type: ignore
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")  # type: ignore
    plt.show()  # type: ignore


def generate_evaluation_dataset(
    llm_model: str, concept: str, class_1: str, class_2: str, num_samples: int
) -> str | None:
    """Generates a synthetic dataset for evaluation."""
    print(f"Generating evaluation dataset with {num_samples} verbatim...")
    PROMPT = evaluation_dataset_generation_prompt(
        concept=concept, class_1=class_1, class_2=class_2, num_samples=num_samples
    )
    client = OpenAI(api_key=f"{os.getenv("OPENAI_API_KEY")}")
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Dataset :"},
    ]
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,  # type: ignore
        temperature=1,
    )
    content = response.choices[0].message.content
    print(f"Dataset generated.")
    return content
