import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Definição dos diretórios contendo as imagens
HEALTHY_IMAGES_DIR = 'healthy'
PROBLEMATIC_IMAGES_DIR = 'problematic'

# Função para carregar imagens de um diretório
def load_images_from_directory(directory, label, img_size=(224, 224)):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter BGR para RGB
            resized_img = cv2.resize(img, img_size)
            images.append(resized_img)
            labels.append(label)
    return images, labels

# Carregamento das imagens e seus rótulos
healthy_images, healthy_labels = load_images_from_directory(HEALTHY_IMAGES_DIR, label=0)
problematic_images, problematic_labels = load_images_from_directory(PROBLEMATIC_IMAGES_DIR, label=1)

# Combinação de todas as imagens e rótulos
all_images = np.array(healthy_images + problematic_images)
all_labels = np.array(healthy_labels + problematic_labels)

# Normalização das imagens (valores entre 0 e 1)
normalized_images = all_images.astype('float32') / 255.0

# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(normalized_images, all_labels, test_size=0.2, random_state=42)

# Conversão dos rótulos para formato categórico
y_train_categorical = to_categorical(y_train, 2)
y_test_categorical = to_categorical(y_test, 2)

# Cálculo dos pesos das classes para lidar com possível desbalanceamento
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Configuração do aumento de dados
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2
)

# Carregamento do modelo pré-treinado VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelamento das camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adição de camadas personalizadas
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

# Criação do modelo final
model = Model(inputs=base_model.input, outputs=output)

# Compilação do modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(
    datagen.flow(X_train, y_train_categorical, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=20,
    validation_data=(X_test, y_test_categorical),
    class_weight=class_weights_dict
)

# Avaliação do modelo
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f'Acurácia do modelo: {accuracy * 100:.2f}%')

# Geração de relatório de classificação
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_categorical, axis=1)
print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred_classes, target_names=['Saudável', 'Problemática']))

# Plotagem do gráfico de acurácia
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treinamento')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o Treinamento e Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Plotagem do gráfico de perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treinamento')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda durante o Treinamento e Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Salvamento dos gráficos
plt.tight_layout()
plt.savefig('graficos_treinamento.png')
plt.close()

# Visualização de ativações (exemplo para a primeira imagem de teste)
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import Model

# Criar um modelo que retorna as ativações da última camada convolucional
last_conv_layer = base_model.get_layer('block5_conv3')
feature_map_model = Model(inputs=base_model.inputs, outputs=last_conv_layer.output)

# Obter o mapa de características para a primeira imagem de teste
img = X_test[0]
img = np.expand_dims(img, axis=0)
feature_map = feature_map_model.predict(img)

# Plotar a imagem original e o mapa de características
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(X_test[0])
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.mean(feature_map[0], axis=-1), cmap='viridis')
plt.title('Mapa de Ativações')
plt.axis('off')

plt.savefig('visualizacao_ativacoes.png')
plt.close()

print("Treinamento concluído. Os gráficos foram salvos como 'graficos_treinamento.png' e 'visualizacao_ativacoes.png'.")