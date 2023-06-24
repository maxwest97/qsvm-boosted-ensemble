from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np

import sys
from c2qa import * # https://github.com/C2QA/bosonic-qiskit/tree/main

def cv_circuit(event, num_qumodes, num_qubits_per_qumode, disp=0, params=None):

    nqumodes = event.shape[0]

    qmr = c2qa.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode)
    qc  = c2qa.CVCircuit(qmr)
	
    for i in range(nqumodes):
        if disp:
            qc.cv_d(event[i], qmr[i])
        else:
            qc.cv_sq(event[i], qmr[i])
	
    return Statevector.from_int(0, 2**(nqumodes * num_qubits_per_qumode)).evolve(qc)

def pca_cv_circuit(event, num_qumodes, num_qubits_per_qumode, params=None):

    nqumodes = event.shape[0]

    qmr = c2qa.QumodeRegister(num_qumodes=num_qumodes, num_qubits_per_qumode=num_qubits_per_qumode)
    qc  = c2qa.CVCircuit(qmr)

    for i in range(nqumodes):
        qc.cv_sq2(event[i], qmr[i], qmr[(i+1) % nqumodes])
    for i in range(nqumodes):
        qc.cv_d(event[i], qmr[i])

    return Statevector.from_int(0, 2**(nqumodes * num_qubits_per_qumode)).evolve(qc)

def original_circuit(event, params, randomise=0):

    nqubits = params.shape[2]
    qc  = QuantumCircuit(nqubits)

    if not(randomise) or (params[0,0,0] < 2*np.pi/3):
        encode_gate = qc.rz
    elif params[0,0,0] < 4*np.pi/3:
        encode_gate = qc.ry
    else:
        encode_gate = qc.rx

    if not(randomise) or (params[0,1,0] < 2*np.pi/3):
        parametrised_gate = qc.ry
    elif params[0,1,0] < 4*np.pi/3:
        parametrised_gate = qc.rz
    else:
        parametrised_gate = qc.rx

    if not(randomise) or (params[0,0,1] < 2*np.pi/3):
        entangling_gate = qc.cry
    elif params[0,0,1] < 4*np.pi/3:
        entangling_gate = qc.crz
    else:
        entangling_gate = qc.crx

    if len(event):
        for j, p in enumerate(params):
            for i in range(nqubits):

                qc.h(i)
                encode_gate(event[(i + j * nqubits) % len(event)], i)
                parametrised_gate(p[0,i], i)
                #qc.rz(event[(i + j * nqubits) % len(event)], i)
                #qc.ry(p[0,i], i)

            for i in range(nqubits):
                #qc.cnot(i, (i+1) % nqubits)
                if nqubits > 1:
                    entangling_gate(p[1,i], i, (i+1) % nqubits)

    return Statevector.from_int(0, 2**nqubits).evolve(qc)

