# CNNs are often supervised models (labeled data used for training), primarily used for image classification

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import numpy as np
import matplotlib.pyplot as plt

# Simulated data (PLEASE replace with real scraped data for any serious application)
headlines = ["Stocks are up today", "Markets crash due to economic instability", "Neutral day in the market"] * 10  # Replicating for more data
labels = [1, 0, 2] * 10  # 1: positive, 0: negative, 2: neutral

# Text Preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(headlines)
vocab_size = len(tokenizer.word_index) + 1
max_length = max([len(s.split()) for s in headlines])

sequences = tokenizer.texts_to_sequences(headlines)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, np.array(labels), test_size=0.2, random_state=42)

# Model Building
model = Sequential()
model.add(Embedding(vocab_size, 16, input_length=max_length))
model.add(Conv1D(16, 3, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.2)

# Testing
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Visualization
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Add performance metrics and explanations to the plot
metrics_text = f'''Test Loss: {loss:.4f} (Lower is better)
Test Accuracy: {accuracy:.4f} (Higher is better)'''

plt.gcf().text(0.02, 0.5, metrics_text, fontsize=12, verticalalignment='center', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

# Add model explanation to the plot
model_explanation = '''This CNN model analyzes financial news headlines to categorize the sentiment as Positive, Negative, or Neutral.
The model is trained on tokenized text data, and uses Conv1D layers to identify local patterns within the text.
After training, the model is evaluated on a separate test set to assess its predictive accuracy.'''

plt.gcf().text(0.6, 0.2, model_explanation, fontsize=6, verticalalignment='center', bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 10})

plt.suptitle("Financial News Sentiment Analysis using CNN")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the plot as a .png file
plt.savefig('Financial_News_Sentiment_Analysis.png')

plt.show()
