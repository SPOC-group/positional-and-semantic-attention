#!/bin/bash

while true; do
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.02 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.04 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.06 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.08 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.1 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.12 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.14 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.16 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.18 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.2 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.22 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.24 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.26 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.28 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.3 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.32 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.34 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.36 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.38 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.4 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.42 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.44 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.46 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.48 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.5 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.55 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.7 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.3 --delta 0.4 --omega 0.3 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.7 --delta 0.4 --omega 0.3 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 1.0 --delta 0.4 --omega 0.3 --N_iter 1 --lmbda 0.01 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.3 --N_iter 1 --lmbda 0.1 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
	python3 -m src.empirics_mixed_teacher_softmax --optim=GD --alphas 0.01 2.0 25 --d 1000 --L 2 --r 1 --sigma 0.5 --delta 0.4 --omega 0.3 --N_iter 1 --lmbda 0.001 --lmbda_linear 0.0001 --exp 01_paper_toy_example_phase_diagram_pos_init --informed_position
done