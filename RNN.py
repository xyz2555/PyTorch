import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# Import PyTorch, modul neural network, dan matplotlib untuk plotting loss

from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, letter_to_tensor, line_to_tensor, random_training_example
# Import utilitas:
# - ALL_LETTERS: semua karakter yang digunakan
# - N_LETTERS: jumlah karakter unik
# - load_data(): load dataset (nama → kategori)
# - letter_to_tensor(): ubah 1 huruf jadi tensor
# - line_to_tensor(): ubah satu kata jadi sequence tensor
# - random_training_example(): ambil sampel training acak

class RNN(nn.Module):
    def __init__(self, input_sizes, hidden_sizes, outputs_sizes):
        super(RNN, self).__init__()
        # Memanggil constructor parent nn.Module

        self.hidden_size = hidden_sizes
        # Menyimpan ukuran hidden state

        self.i2h = nn.Linear(input_sizes + hidden_sizes, hidden_sizes)
        # Linear layer untuk menghitung hidden state baru
        # Input = [input_t, hidden_t] digabung (concatenate)

        self.i2o = nn.Linear(input_sizes + hidden_sizes, outputs_sizes)
        # Linear layer untuk menghasilkan output (logits kategori)

        self.softmax = nn.LogSoftmax(dim=1)
        # LogSoftmax untuk menghasilkan log-probability
        # Digunakan bersama NLLLoss

    def forward(self, input_tensor, hidden_tensor):
        # Forward RNN satu time-step

        combined = torch.cat((input_tensor, hidden_tensor), 1)
        # Menggabungkan input huruf dan hidden state sebelumnya

        hidden = self.i2h(combined)
        # Menghitung hidden state baru

        output = self.i2o(combined)
        # Menghasilkan output logits

        output = self.softmax(output)
        # Mengubah logits menjadi log-probability

        return output, hidden
        # Mengembalikan output dan hidden state terbaru
    
    def init_hidden(self):
        # Inisialisasi hidden state awal
        return torch.zeros(1, self.hidden_size)
        # Hidden state awal = nol

category_lines, all_categories = load_data()
# Load dataset:
# - category_lines: dict {kategori: [list nama]}
# - all_categories: list semua kategori

n_categories = len(all_categories)
# Jumlah kelas / kategori
# print(n_categories)

n_hidden = 128
# Ukuran hidden state RNN

rnn = RNN(N_LETTERS, n_hidden, n_categories)
# Membuat model RNN
# Input  = jumlah huruf unik
# Hidden = 128
# Output = jumlah kategori

input_tensor = letter_to_tensor('A')
# Mengubah huruf 'A' menjadi tensor one-hot

hidden_tensor = rnn.init_hidden()
# Inisialisasi hidden state

output, next_hidden = rnn(input_tensor, hidden_tensor)
# Forward pass satu huruf
# print(output.size())
# print(next_hidden.size())

input_tensor = line_to_tensor('Albert')
# Mengubah kata "Albert" menjadi sequence tensor

hidden_tensor = rnn.init_hidden()
# Reset hidden state

output, next_hidden = rnn(input_tensor[0], hidden_tensor)
# Forward pass huruf pertama saja (contoh)
# print(output.size())
# print(next_hidden.size())

def category_from_output(output):
    # Mengambil kategori dengan probabilitas tertinggi
    category_idx = torch.argmax(output).item()
    # Ambil index nilai terbesar
    return all_categories[category_idx]
    # Kembalikan nama kategori

print(category_from_output(output))
# Menampilkan prediksi kategori sementara

criterion = nn.NLLLoss()
# Loss function:
# Negative Log Likelihood Loss
# Cocok untuk output LogSoftmax

learning_rate = 0.005
# Learning rate

optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
# Optimizer SGD untuk update parameter RNN

def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    # Reset hidden state untuk setiap nama baru

    for i in range(line_tensor.size()[0]):
        # Loop setiap huruf dalam nama
        output, hidden = rnn(line_tensor[i], hidden)
        # Forward RNN step-by-step (sequence processing)

    loss = criterion(output, category_tensor)
    # Hitung loss berdasarkan output terakhir

    optimizer.zero_grad()
    # Reset gradien

    loss.backward()
    # Backpropagation Through Time (BPTT)

    optimizer.step()
    # Update parameter model

    return output, loss.item()
    # Return output dan nilai loss

current_loss = 0.0
all_losses = []
# Variabel untuk menyimpan loss rata-rata

plot_steps, print_steps = 1000, 5000
# Interval plotting dan printing

n_iters = 100000
# Jumlah iterasi training

for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(
        category_lines, all_categories
    )
    # Ambil contoh training acak

    output, loss = train(line_tensor, category_tensor)
    # Training satu data

    current_loss += loss
    # Akumulasi loss

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        # Simpan rata-rata loss
        current_loss = 0
    
    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        # Prediksi kategori

        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        # Cek benar atau salah

        print(f'{i} {i/n_iters*100} {loss:.4f} {line} / {guess} {correct}')
        # Print progress training

plt.figure()
plt.plot(all_losses)
plt.show()
# Plot grafik loss selama training

def predict(input_line):
    # Fungsi prediksi untuk input user
    print(f"\n> {input_line}")

    with torch.no_grad():
        # Disable gradient (mode inference)

        line_tensor = line_to_tensor(input_line)
        # Convert input ke tensor sequence

        hidden = rnn.init_hidden()
        # Reset hidden state

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
            # Forward RNN per huruf

        guess = category_from_output(output)
        # Ambil prediksi akhir

        print(guess)
        # Tampilkan hasil

while True:
    sentence = input("Input:")
    # Input dari user

    if sentence == "quit":
        break
        # Keluar dari program

    predict(sentence)
    # Jalankan prediksi

# ============================================================
# RINGKASAN KESELURUHAN CODE
#
# Kode ini membangun dan melatih Recurrent Neural Network (RNN)
# untuk klasifikasi nama berdasarkan urutan huruf.
#
# Alur utama:
# 1) Dataset berisi nama → kategori (misalnya negara asal)
# 2) Setiap nama diproses sebagai sequence karakter
# 3) RNN membaca huruf satu per satu sambil menyimpan hidden state
# 4) Output terakhir digunakan untuk menentukan kategori
# 5) Loss menggunakan NLLLoss + LogSoftmax
# 6) Training dilakukan dengan Backpropagation Through Time (BPTT)
# 7) Loss diplot untuk memantau proses belajar
# 8) Model dapat digunakan untuk prediksi interaktif via input user
#
# Intinya:
# Model ini belajar POLA URUTAN karakter,
# bukan hanya frekuensi huruf, sehingga cocok
# untuk data sequence seperti teks atau time-series.
# ============================================================
