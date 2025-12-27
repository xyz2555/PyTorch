import torch                          # Library utama PyTorch
import torch.nn as nn                 # Modul neural network (Linear, Loss, dll)
import torchvision                    # Dataset dan utilitas vision
import torchvision.transforms as transforms  # Transformasi data gambar
import matplotlib.pyplot as plt       # Visualisasi gambar
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logger
import sys                            # Untuk exit program (opsional)
import torch.nn.functional as F       # Functional API (softmax, relu, dll)

writer = SummaryWriter("runs/mnist2")
# Membuat writer TensorBoard
# Semua log akan disimpan di folder runs/mnist2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Menentukan device komputasi:
# - Gunakan GPU jika tersedia
# - Jika tidak, gunakan CPU

input_size = 784
# Ukuran input MNIST:
# 28 x 28 piksel → di-flatten menjadi 784

hidden_size = 100
# Jumlah neuron pada hidden layer (hyperparameter)

num_classes = 10
# Jumlah kelas output (digit 0 sampai 9)

num_epochs = 2
# Jumlah epoch training

batch_size = 100
# Jumlah data dalam satu batch

learning_rate = 0.01
# Learning rate awal (nanti di-override)

train_dataset = torchvision.datasets.MNIST(
    root='./data',                   # Folder penyimpanan dataset
    train=True,                      # Dataset training
    transform=transforms.ToTensor(), # Konversi ke tensor [0,1]
    download=True                   # Download jika belum ada
)

test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,                     # Dataset testing
    transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,           # Dataset training
    batch_size=batch_size,           # Ukuran batch
    shuffle=True                     # Data diacak setiap epoch
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False                    # Testing tidak perlu shuffle
)

example = iter(train_loader)
# Membuat iterator dari DataLoader

samples, labels = next(example)
# Mengambil satu batch contoh data

print(samples.shape, labels.shape)
# samples: [batch_size, 1, 28, 28]
# labels : [batch_size]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap='gray')
# Menampilkan 6 contoh gambar MNIST

img_grid = torchvision.utils.make_grid(samples)
# Menggabungkan batch gambar menjadi satu grid

writer.add_image('mnist image', img_grid)
# Menyimpan grid gambar ke TensorBoard

writer.close()
# Menutup writer (flush data)

# ======================
# MODEL MLP
# ======================

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        # Fully Connected layer: input → hidden

        self.relu = nn.ReLU()
        # Fungsi aktivasi non-linear

        self.l2 = nn.Linear(hidden_size, num_classes)
        # Output layer: hidden → logits kelas

    def forward(self, x):
        out = self.l1(x)              # Transformasi linear
        out = self.relu(out)          # Aktivasi ReLU
        out = self.l2(out)            # Output logits
        return out                    # Softmax ditangani oleh loss

model = NeuralNet(input_size, hidden_size, num_classes)
# Membuat instance model

learning_rate = 0.001
# Learning rate aktual (mengganti nilai sebelumnya)

criterion = nn.CrossEntropyLoss()
# Loss function:
# Sudah termasuk LogSoftmax + NLLLoss
# Mengharapkan logits mentah

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Optimizer Adam untuk update bobot

writer.add_graph(model, samples.reshape(-1, 28*28))
# Menyimpan graph model ke TensorBoard
# Input dummy harus sesuai shape input model

writer.close()

n_total_steps = len(train_loader)
# Jumlah batch per epoch

running_loss = 0.0
running_correct = 0.0
# Variabel akumulasi untuk logging TensorBoard

for epoch in range(num_epochs):
    # Loop epoch
    for i, (images, labels) in enumerate(train_loader):
        # Loop batch

        images = images.reshape(-1, 28*28).to(device)
        # Flatten gambar dan pindahkan ke device

        labels = labels.to(device)
        # Pindahkan label ke device

        outputs = model(images)
        # Forward pass → logits

        loss = criterion(outputs, labels)
        # Hitung loss

        optimizer.zero_grad()
        # Reset gradien

        loss.backward()
        # Backpropagation

        optimizer.step()
        # Update bobot

        running_loss += loss.item()
        # Akumulasi loss

        _, predicted = torch.max(outputs.data, 1)
        # Ambil kelas dengan logit terbesar

        running_correct += (predicted == labels).sum().item()
        # Hitung prediksi benar

        if (i+1) % 100 == 0:
            # Logging setiap 100 batch

            print(f'epoch: {epoch+1} / {num_epochs}, step: {i+1} / {n_total_steps}, loss: {loss.item():.4f}')

            writer.add_scalar(
                'training_loss',
                running_loss / 100,
                epoch * n_total_steps + i
            )
            # Log loss ke TensorBoard

            writer.add_scalar(
                'accuracy',
                running_correct / 100,
                epoch * n_total_steps + i
            )
            # Log akurasi batch ke TensorBoard

            running_loss = 0.0
            running_correct = 0.0
            # Reset akumulasi

labels = []
preds = []
# Menyimpan prediksi untuk PR curve

with torch.no_grad():
    # Mode evaluasi (tanpa gradien)

    n_correct = 0
    n_samples = 0

    for images, labels1 in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels1 = labels1.to(device)

        outputs = model(images)
        # Forward pass

        _, predicted = torch.max(outputs.data, 1)
        # Prediksi kelas

        n_samples += labels1.shape[0]
        n_correct += (predicted == labels1).sum().item()
        # Hitung akurasi

        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        # Softmax per sample (untuk PR curve)

        preds.append(class_predictions)
        labels.append(predicted)

    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels)

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')
    # Akurasi akhir testing

    classes = range(10)
    for i in classes:
        labels_i = labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        # Precision-Recall curve per kelas

        writer.close()

# =====================================================================
# RINGKASAN:
# Kode ini melatih MLP sederhana untuk klasifikasi MNIST dan
# mengintegrasikannya dengan TensorBoard.
#
# Fitur utama:
# - Logging gambar, graph model, loss, accuracy
# - Training berbasis batch dan epoch
# - Evaluasi akurasi pada test set
# - Visualisasi Precision-Recall Curve per kelas
#
# Model menghasilkan logits (tanpa softmax eksplisit),
# dan CrossEntropyLoss menangani softmax secara internal.
# =====================================================================
