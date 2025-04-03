import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# 1
from classifier_model.CNN_LSTM_dataset import HelplessnessVideoDataset, TransformableSequenceSubset
from classifier_model.CNN_LSTM_model import HelplessnessClassifier

# 2
# from classifier_model.dataset import HelplessnessVideoDataset
# from classifier_model.cnn_3d_model.model import HelplessnessClassifier

# 3
# from classifier_model.dataset import HelplessnessVideoDataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dir = "data/val"
    val_base = HelplessnessVideoDataset(val_dir)

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_dataset = TransformableSequenceSubset(val_base, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = HelplessnessClassifier()
    model.load_state_dict(torch.load('classifier_model/grayscale_cnn_lstm.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()

    def validate():
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                _, preds = torch.max(outputs, 1)

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())

        return true_labels, pred_labels

    # Create confusion matrix
    true_labels, pred_labels = validate()

    cm = sk_confusion_matrix(true_labels, pred_labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['None', 'Little', 'Extreme'])
    cm_display.plot(cmap='Blues')
    plt.title("Confusion Matrix for Helplessness Dataset")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.show()

    print(f"Precision score: {precision_score(true_labels, pred_labels, average='weighted'):.2f}")
    print(f"Recall score: {recall_score(true_labels, pred_labels, average='weighted'):.2f}")
    print(f"F1 score: {f1_score(true_labels, pred_labels, average='weighted'):.2f}")
    print(f"Accuracy score: {accuracy_score(true_labels, pred_labels):.2f}")

if __name__ == "__main__":
    main()
