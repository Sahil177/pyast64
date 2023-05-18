import sys
import numpy as np

sys.path.append("..")

from convert import *

from weights import *

file = "test_mlp.p64"


def generate_dataset_quad(num_points, a, b, c):
    inputs = []
    labels = [0] * num_points
    for i in range(num_points):
        input1 = 20 * (np.random.random() - 0.5)
        input2 = 200 * (np.random.random() - 0.5)
        inputs.append(input1)
        inputs.append(input2)
        if a * input1**2 + b * input1 + c > input2:
            labels[i] = 1

    return inputs, labels


with open(file) as f:
    source = f.read()
node = ast.parse(source, filename=file)
compiler = Compiler(peephole=True)
compiler.compile(node)
program, stack_len = convert(compiler.asm.total)
stack_len = 5000
print("s", stack_len)
# inp_start = registers + 8 * stack_len
inp_start = registers + stack_len


print("version 1")
# print(*program, sep="\n")
program = convert_for_state(program)
for i in range(len(program)):
    print(i, program[i])
# print(*program, sep="\n")
input_dim = 2
hidden_dim = 10
output_dim = 1
num_points = 100
a = 1
b = 1
c = -20

test_inps, test_labels = generate_dataset_quad(num_points, a, b, c)

W1_ = []

for i in range(len(W1)):
    for j in range(len(W1[0])):
        W1_.append(W1[i][j])

W2_ = []

for i in range(len(W2)):
    for j in range(len(W2[0])):
        W2_.append(W2[i][j])

W3_ = []

for i in range(len(W3)):
    for j in range(len(W3[0])):
        W3_.append(W3[i][j])

inputs = [
    [num_points],
    test_inps,
    test_labels,
    [input_dim],
    [hidden_dim],
    [output_dim],
    W1_,
    b1,
    W2_,
    b2,
    W3_,
    b3[0],
    [0],
]

lens = [len(lst) for lst in inputs]

# input, os, oe = mlp_process(x,W1, W2, inp_start)
input, os, oe = process(inputs, inp_start)
# print("OG", input)
# lens = [len(lst) for lst in inputs]
os = inp_start + len(inputs) + 1 + sum(lens[-1:])
oe = inp_start + len(inputs) + 1 + sum(lens)
print(os)
print(inp_start)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print(out)
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)
