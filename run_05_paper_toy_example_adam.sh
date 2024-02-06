#!/bin/bash

while true; do
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 2.0 2.0 1 --d 100 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.3 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp mixed_teacher_softmax_adam_d=100 --optim=Adam
done