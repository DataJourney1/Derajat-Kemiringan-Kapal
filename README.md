# Derajat-Kemiringan-Kapal
Latihan melihat derajat kemiringan kapal dengan menggunakan Fuzzy Inference System
```
#Install Library Python
pip install scikit-fuzzy
import numpy as np
import math
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import matplotlib

#membuat rentangan nilai keanggotaan fuzzy
kemiringan = np.arange(0,17.5,0.5)
durasi = np.arange(0,60,0.5)

#membuat membership function
#membership function kemiringankapal
kemiringan_stabil = fuzz.trapmf(kemiringan, [0, 0, 4.5, 5])
kemiringan_miring = fuzz.trapmf(kemiringan, [4.5, 5, 9.5, 10])
kemiringan_sangatmiring = fuzz.trapmf(kemiringan, [9.5, 10, 15, 15])


#membership function durasi kemiringan
durasi_sangatcepat = fuzz.trapmf(durasi, [0, 1,5,6])
durasi_cepat = fuzz.trapmf(durasi, [5, 6, 10, 11])
durasi_cukupcepat = fuzz.trapmf(durasi, [10, 11, 15,16])
durasi_lambat = fuzz.trapmf(durasi, [15, 16, 30,31])
durasi_sangatlambat = fuzz.trapmf(durasi, [30,31, 59,60])

#membership function status kapal dengan singleton
Berbahaya= 0.25
Waspada= 0.50
Aman = 0.75


# Visualize these universes and membership functions
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 8))

ax0.plot(kemiringan, kemiringan_stabil, 'g', linewidth=1.5, label='stabil')
ax0.plot(kemiringan, kemiringan_miring, 'y', linewidth=1.5, label='miring ')
ax0.plot(kemiringan, kemiringan_sangatmiring, 'r', linewidth=1.5, label='Sangat miring')

ax0.set_title('Kemiringan Kapal')
ax0.legend()

ax1.plot(durasi, durasi_sangatcepat, 'g', linewidth=1.5, label='Sangat Cepat')
ax1.plot(durasi, durasi_cepat, 'b', linewidth=1.5, label='Cepat')
ax1.plot(durasi, durasi_cukupcepat, 'y', linewidth=1.5, label='Cukup Cepat')
ax1.plot(durasi, durasi_lambat, 'r', linewidth=1.5, label='Lambat')
ax1.plot(durasi, durasi_sangatlambat, 'black', linewidth=1.5, label='Sangat Lambat')

ax1.set_title('Durasi Kemiringan')
ax1.legend()

#menampilkan singeleton miring
ax2.plot([1, 1], [0, Aman], 'g', linewidth=1.5, label='AMAN (A)')
ax2.plot([2, 2], [0, Waspada], 'y', linewidth=1.5, label='WASPADA (W)')
ax2.plot([3, 3], [0, Berbahaya], 'r', linewidth=1.5, label='BERBAHAYA (B)')

ax2.set_title('Status Kapal Miring')
ax2.legend()
# Turn off top/right axes
for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

#input kemiringan kapal dan durasi
input_kemiringan = 4.8
#input Durasi Kemiringan
input_durasi = 10

if input_kemiringan < 0 :
    input_kem = input_kemiringan *(-1)
    input_kemiringan = input_kem

# Input Aturan Yang sudah didapatkan dari Pakar
# dan input bobot singeleton outputnya
A=0.75;W=0.50;B=0.25
aturan = [[A,A,A,A,A],[W,W,W,W,B],[B,B,B,B,B]]

# Fuzzification
input_1 = []
input_1.append(
    fuzz.interp_membership(
        kemiringan, kemiringan_stabil, input_kemiringan))
input_1.append(
    fuzz.interp_membership(
        kemiringan, kemiringan_miring, input_kemiringan))
input_1.append(
    fuzz.interp_membership(
        kemiringan, kemiringan_sangatmiring, input_kemiringan))

input_2 = []
input_2.append(
    fuzz.interp_membership(
        durasi, durasi_sangatcepat, input_durasi))
input_2.append(
    fuzz.interp_membership(
        durasi, durasi_cepat, input_durasi))
input_2.append(
    fuzz.interp_membership(
        durasi, durasi_cukupcepat, input_durasi))
input_2.append(
    fuzz.interp_membership(
        durasi, durasi_lambat, input_durasi))
input_2.append(
    fuzz.interp_membership(
        durasi, durasi_sangatlambat, input_durasi))

print("Derajat Keanggotaan Kemiringan kapal:")

if input_1[0]>0 :
    print("Stabil            :"+ str(input_1[0]))

if input_1[1]>0 :
    print("Miring      :"+ str(input_1[1]))

if input_1[2]>0 :
    print("Sangat Miring:"+ str(input_1[2]))

print("")
print("Derajat Keanggotaan Durasi Kemiringan:")
if input_2[0]>0 :
    print("Sangat Cepat    :"+ str(input_2[0]))
if input_2[1]>0 :
    print("Cepat   :"+ str(input_2[1]))
if input_2[2]>0 :
    print("Cukup Cepat  :"+ str(input_2[2]))
if input_2[3]>0 :
    print("Lambat  :"+ str(input_2[3]))
if input_2[4]>0 :
    print("Sangat Lambat  :"+ str(input_2[4]))


print("Matriks Kemiringan Kapal:")
print(input_1 )
print("")
print("Matriks Durasi Kemiringan:")
print(input_2)  

# Inference dan Defazzification dengan Centroid Sugeno:
#Penyebut
if input_kemiringan >= 0:
    rul = []
    for i in range(3):
        for j in range(5):
            rule=fuzz.relation_min(input_1[i], input_2[j])
            #fungsi append untuk menambahkan file baru
            #(rule),kedalam data list rul
            rul.append(rule)

    penyebut=np.sum(rul)
    #Pembilang
    print("------------------------")
    rul = []
    for i in range(3):
        for j in range(5):
            #mengambil nilai MIN dalam aturan yang terpicu
            rule=fuzz.relation_min(input_1[i], input_2[j])
            rulxx= rule*aturan[i][j]
             #fungsi append untuk menambahkan file baru
            #(rulxx),kedalam data list rul
            rul.append(rulxx)

    pembilang=np.sum(rul)
    # menghitung nilai crips metode centroid trapesium
    hasil = pembilang/penyebut



    ## Logic untuk hasil
    print("Kemiringan :"+ str(input_kemiringan))
    print("Durasi : "+ str(input_durasi))
    if hasil>=0.25 and hasil<= 0.50:
        print("Status Kapal :"+ str(hasil))
        print("BERBAHAYA")

    if hasil>0.50 and hasil< 0.75:
        print("Status Kapal :"+ str(hasil))
        print("WASPADA")

    if hasil>=0.75 and hasil <= 1:
        print("Status Kapal :"+ str(hasil))
        print("AMAN")
