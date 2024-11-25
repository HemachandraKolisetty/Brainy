import torch
import time

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=20, model_type='snn'):
    best_val_accuracy = 0.0
    best_model_state = None

    # track loss and accuracies
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        start_time = time.time()
        avg_loss = 0
        total_correct = 0
        total_samples = 0
        for data, targets in train_loader:
            data = data.float()

            # Forward pass
            spk_rec = model(data)
            spk_sum = spk_rec.sum(dim=0) if model_type == 'snn' else spk_rec

            if criterion.__class__.__name__ == 'CrossEntropyLoss':
                targets = targets.long()
                loss = criterion(spk_sum, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                avg_loss += loss.item()
                _, predicted = spk_sum.max(1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
            else:
                # normalize spk_sum
                num_steps = data.size(1)
                spk_sum = spk_sum / num_steps
                loss = criterion(spk_sum, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                avg_loss += loss.item()
                _, predicted = spk_sum.max(1)
                total_correct += (predicted == targets.argmax(dim=1)).sum().item()
                total_samples += targets.size(0)
            

        train_acc = total_correct / total_samples
        train_loss = avg_loss / len(train_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation Phase
        model.eval()
        val_correct = 0
        val_samples = 0
        val_loss = 0.0
        with torch.no_grad():
            for val_data, val_targets in val_loader:
                val_data = val_data.float()
                val_targets = val_targets.long()

                # Forward pass
                spk_rec_val = model(val_data)
                spk_sum_val = spk_rec_val.sum(dim=0) if model_type == 'snn' else spk_rec_val
                val_loss += criterion(spk_sum_val, val_targets).item()

                # Metrics
                _, val_predicted = spk_sum_val.max(1)

                if criterion.__class__.__name__ == 'CrossEntropyLoss':
                    val_correct += (val_predicted == val_targets).sum().item()
                else:
                    val_correct += (val_predicted == val_targets.argmax(dim=1)).sum().item()
                val_samples += val_targets.size(0)

        val_acc = val_correct / val_samples
        val_loss = val_loss / len(val_loader)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Check for best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_state = model.state_dict()
        
        elapsed_time = time.time() - start_time

        # Log metrics
        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc*100:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%, "
            f"Time: {elapsed_time:.2f}s"
        )

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Best model loaded with Validation Accuracy: {best_val_accuracy*100:.2f}%")

    return train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model, test_loader, encoding='label', model_type='snn'):
    model.eval()
    total_correct = 0
    total_samples = 0
    start_time = time.time()
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.float()
            targets = targets.long()
            spk_rec = model(data)
            spk_sum = spk_rec.sum(dim=0) if model_type == 'snn' else spk_rec
            _, predicted = spk_sum.max(1)
            if encoding == 'label':
                total_correct += (predicted == targets).sum().item()
            else:
                total_correct += (predicted == targets.argmax(dim=1)).sum().item()
            total_samples += targets.size(0)
    acc = total_correct / total_samples
    elapsed_time = time.time() - start_time
    print(f"Test Accuracy: {acc*100:.2f}%, Time: {elapsed_time:.2f}s")
    return acc