import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import cv2

def build(n, d):
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.h(i)
    for layer in range(d):
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.rz(np.pi / (layer + 2), i)
        for i in range(n):
            qc.ry(np.pi / (i + 2), i)
    qc.measure_all()
    return qc

def run(qc, shots):
    sim = AerSimulator()
    job = sim.run(qc, shots=shots)
    res = job.result()
    counts = res.get_counts()
    return counts

def to_phase(counts, h, w):
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}
    vals = np.array(list(probs.values()))
    flat = np.zeros(h * w)
    for i in range(h * w):
        flat[i] = vals[i % len(vals)]
    np.random.shuffle(flat)
    mn, mx = flat.min(), flat.max()
    if mx - mn > 0:
        flat = (flat - mn) / (mx - mn)
    img = flat.reshape(h, w)
    img = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 2)
    mn, mx = img.min(), img.max()
    if mx - mn > 0:
        img = (img - mn) / (mx - mn)
    return img