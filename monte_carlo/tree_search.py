import sys

sys.path.append("..")

from convert import *

file = "tree_search.p64"

from weights import *

np.random.seed(0)


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
stack_len = 5000000
print("s", stack_len)
# inp_start = registers + 8 * stack_len
inp_start = registers + stack_len


print("version 1")
# print(*program, sep="\n")
program = convert_for_state(program)
for i in range(len(program)):
    print(i, program[i])
# print(*program, sep="\n")
num_points = 100
input_dim = 2
hidden_dim = 10
output_dim = 1
x = np.random.randint(low=-10, high=10, size=(num_points, input_dim))
# W1 = np.random.randint(low=-10, high=10, size=(hidden_dim,input_dim))
# W2 = np.random.randint(low=-10, high=10, size =(output_dim,hidden_dim))

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


inps = [[]]
inps = [
    [0, 1, 2, 4, 5, 6, 11],
    [0, 2, 11, 12],
    [0, 2, 2, 12, 7, 8, 13],
    [0, 2, 13, 14],
    [0, 2, 3, 14, 9, 10, 15],
]

Ws = []
for i in range(len(inps)):
    inp = inps[i]
    W = np.eye(16, dtype=np.int64)
    for j in range(len(inp)):
        W[j] = 0
        W[j, inp[j]] = 1

    Ws.append(W)

ord = [
    78,
    0,
    0,
    0,
    0,
    0,
    0,
    39,
    0,
    0,
    71,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    57,
    0,
    0,
    0,
    0,
    99,
]

inputs = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0] * 100,
    [0],
    test_labels,
    [0],
    [16],
    [num_points],
    [input_dim],
    [hidden_dim],
    [hidden_dim],
    test_inps,
    W1_,
    b1,
    W2_,
    b2,
    W3_,
    b3[0],
    [0] * hidden_dim * num_points,
    [0] * hidden_dim * num_points,
    [0] * hidden_dim * num_points,
    [0] * hidden_dim * num_points,
    [0] * hidden_dim * num_points,
    [1],  # timesteps
    [2],  # num skills
    list(Ws[0].flatten()),
    list(Ws[1].flatten()),
    list(Ws[2].flatten()),
    list(Ws[3].flatten()),
    list(Ws[4].flatten()),
    [1, 0],
    [0, 1],
]

# inputs = [
#     test_labels,
#     [0],
#     [16],
#     [num_points],
#     [input_dim],
#     [hidden_dim],
#     [output_dim],
#     test_inps,
#     W1_,
#     b1,
#     W2_,
#     b2,
#     W3_,
#     b3[0],
#     [0] * hidden_dim * num_points,
#     [0] * hidden_dim * num_points,
#     [0] * hidden_dim * num_points,
#     [0] * hidden_dim * num_points,
#     [0] * output_dim * num_points,
#     [5],  # timesteps
#     [2],  # num skills
#     list(Ws[0].flatten()),
#     list(Ws[1].flatten()),
#     list(Ws[2].flatten()),
#     list(Ws[3].flatten()),
#     list(Ws[4].flatten()),
#     [1, 0],
#     [0, 1],
#     [1, 0],
#     [0, 1],
#     [1, 0],
# ]

lens = [len(lst) for lst in inputs]

# input, os, oe = mlp_process(x,W1, W2, inp_start)
input, os, oe = process(inputs, inp_start)
# print("OG", input)
# lens = [len(lst) for lst in inputs]
os = inp_start + len(inputs) + 1 + sum(lens[:2])
oe = inp_start + len(inputs) + 1 + sum(lens[:4])
print(inp_start)
state = execute(program, stack_len, input)
# out = process_out(state[os:oe])
out = state[os:oe]
print(out)
# acc = test(100, 10 ,10, inp_start, program, stack_len)
# print(acc)
