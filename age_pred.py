import os
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from utils.tcav import run_tcav
from torch.utils.data import DataLoader
from utils.model import create_resnet18
from utils.dataset import CelebADataset
import torchvision.transforms as transforms
from utils.data_split import load_partitioned_data
from utils.data_annotate import load_data_annotations
from sklearn.metrics import classification_report, precision_score, recall_score

os.makedirs('./logs', exist_ok=True)
os.makedirs('./models/age', exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, model_path, model_name, device, test_loader, log_file):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f'Testing {model_name}', unit='batch'):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total * 100
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)

    report = classification_report(
        all_labels_tensor.numpy(),
        all_preds_tensor.numpy(),
        target_names=['Old', 'Young'],
        zero_division=0
    )

    print(f"\n--- Test Results for {model_name} ---", file=log_file)
    print(f"Test Accuracy: {test_acc:.2f}%", file=log_file)
    print(report, file=log_file)

def main():
    set_seed(42)
    with open('./logs/age.txt', 'w') as log_file:
        train_paths, validation_paths, test_paths = load_partitioned_data()
        train_labels, validation_labels, test_labels = load_data_annotations(attribute="Young")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = CelebADataset(train_paths, train_labels, transform=transform)
        val_dataset = CelebADataset(validation_paths, validation_labels, transform=transform)
        test_dataset = CelebADataset(test_paths, test_labels, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('Using GPU', file=log_file)
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Using Metal Performance Shaders (Apple)\n', file=log_file)
        else:
            device = torch.device('cpu')
            print('Using CPU', file=log_file)

        model = create_resnet18(num_classes=1).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 10

        best_val_acc = 0.0
        best_val_precision = 0.0
        best_val_recall = 0.0

        best_acc_epoch = 0
        best_precision_epoch = 0
        best_recall_epoch = 0

        for epoch in tqdm(range(num_epochs), desc='Epoch Progress', unit='epoch'):
            model.train()
            total_loss = 0

            for images, labels in tqdm(train_loader, desc='Training', unit='batch', leave=False):
                images, labels = images.to(device), labels.to(device).unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            val_preds = []
            val_labels_list = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc='Validation', unit='batch', leave=False):
                    images, labels = images.to(device), labels.to(device).unsqueeze(1)
                    outputs = model(images)
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    val_preds.append(predictions.cpu())
                    val_labels_list.append(labels.cpu())

            val_preds = torch.cat(val_preds).numpy()
            val_labels_list = torch.cat(val_labels_list).numpy()

            val_acc = (val_preds == val_labels_list).mean() * 100
            val_precision = precision_score(val_labels_list, val_preds)
            val_recall = recall_score(val_labels_list, val_preds)

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, "
                f"Val Acc: {val_acc:.2f}%, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}",
                file=log_file
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "./models/age/young_accuracy_model.pth")
                best_acc_epoch = epoch + 1

            if val_precision > best_val_precision:
                best_val_precision = val_precision
                torch.save(model.state_dict(), "./models/age/young_precision_model.pth")
                best_precision_epoch = epoch + 1

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                torch.save(model.state_dict(), "./models/age/young_recall_model.pth")
                best_recall_epoch = epoch + 1

        print(f"\nHighest accuracy saved at epoch {best_acc_epoch} with Val Acc: {best_val_acc:.2f}%", file=log_file)
        print(f"Highest precision saved at epoch {best_precision_epoch} with Val Precision: {best_val_precision:.2f}", file=log_file)
        print(f"Highest recall saved at epoch {best_recall_epoch} with Val Recall: {best_val_recall:.2f}\n", file=log_file)

        evaluate_model(model, "./models/age/young_accuracy_model.pth", "Best Accuracy Model", device, test_loader, log_file)
        evaluate_model(model, "./models/age/young_precision_model.pth", "Best Precision Model", device, test_loader, log_file)
        evaluate_model(model, "./models/age/young_recall_model.pth", "Best Recall Model", device, test_loader, log_file)

if __name__ == '__main__':
    main()
    run_tcav(target_attr = 'Young', model_path = './models/age/young_accuracy_model.pth', target_class = 0, attr_file_path = './data/list_attr_celeba.txt', image_dir = './data/img_align_celeba/', max_images = 100, num_random_set = 2, n_steps = 10, processes = 0, save_plots = True)