import json
import matplotlib.pyplot as plt

with open('training_history.json', 'r') as f:
    history_dict = json.load(f)

epochs = len(history_dict['accuracy'])
epoch_numbers = range(1, epochs + 1)

plt.figure(figsize=(6, 5))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(epoch_numbers, history_dict['accuracy'], label='Training Accuracy')
plt.plot(epoch_numbers, history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epoch_numbers)
plt.legend()

plt.tight_layout()
plt.show()