import matplotlib.pyplot as plt
import pandas as pd

from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, Bidirectional, Dense


max_features = 10000
maxlen = 500
batch_size = 128
epochs = 5


(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)

input_train = pad_sequences(input_train, maxlen=maxlen)
input_test = pad_sequences(input_test, maxlen=maxlen)


def build_simple_rnn():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(SimpleRNN(32))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    return model


def build_lstm():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    return model


def build_bidirectional_lstm():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    return model


def build_deep_bidirectional_lstm():
    model = Sequential()
    model.add(Embedding(max_features, 32))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    return model


models = {
    "SimpleRNN": build_simple_rnn(),
    "LSTM": build_lstm(),
    "Bidirectional LSTM": build_bidirectional_lstm(),
    "Deep Bidirectional LSTM": build_deep_bidirectional_lstm()
}


results = []
histories = {}

for model_name, model in models.items():
    print("\n" + "=" * 50)
    print(f"Навчання моделі: {model_name}")
    print("=" * 50)

    model.summary()

    history = model.fit(
        input_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.3
    )

    test_loss, test_acc = model.evaluate(input_test, y_test)

    results.append({
        "Модель": model_name,
        "Test Loss": test_loss,
        "Test Accuracy": test_acc
    })

    histories[model_name] = history


results_df = pd.DataFrame(results)
print("\nПорівняння результатів:")
print(results_df)


plt.figure(figsize=(10, 6))

for model_name, history in histories.items():
    val_acc = history.history["val_acc"]
    plt.plot(range(1, len(val_acc) + 1), val_acc, label=model_name)

plt.title("Порівняння точності моделей на валідаційних даних")
plt.xlabel("Епоха")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))

for model_name, history in histories.items():
    val_loss = history.history["val_loss"]
    plt.plot(range(1, len(val_loss) + 1), val_loss, label=model_name)

plt.title("Порівняння втрат моделей на валідаційних даних")
plt.xlabel("Епоха")
plt.ylabel("Validation Loss")
plt.legend()
plt.show()


best_model = results_df.loc[results_df["Test Accuracy"].idxmax()]

print("\nВисновок:")
print(f"Найкращий результат показала модель: {best_model['Модель']}")
print(f"Точність на тестових даних: {best_model['Test Accuracy']:.4f}")
